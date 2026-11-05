"""Reference solutions, kept out of the public branch as an encrypted archive.

A benchmark whose answers are public stops measuring anything once the
answers reach training data. The reference solutions of the tasks
(``tasks/<task>/reference/``) are therefore not committed: they are packed
into ``benchmark/references.enc``, encrypted with a key that maintainers keep
outside the repository (``EPX_BENCHMARK_KEY``). Tasks whose ``meta.yaml`` says
``public_reference: true`` (the self-test) keep their reference in the clear.

The cipher uses only the standard library: a keystream of HMAC-SHA256 blocks
(counter mode) XORed with a gzipped tar of the references, authenticated by a
second HMAC over the nonce and ciphertext. The key is stretched with
PBKDF2-HMAC-SHA256.
"""

import hashlib
import hmac
import io
import os
import tarfile
from pathlib import Path

MAGIC = b"EPXREF1\n"
KEY_ENV = "EPX_BENCHMARK_KEY"
_SALT = b"epydemix-benchmark-references"
_ITERATIONS = 200_000


class ReferenceError(RuntimeError):
    pass


def _keys(secret: str):
    master = hashlib.pbkdf2_hmac("sha256", secret.encode(), _SALT, _ITERATIONS)
    return (
        hmac.new(master, b"encrypt", hashlib.sha256).digest(),
        hmac.new(master, b"authenticate", hashlib.sha256).digest(),
    )


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    blocks = []
    for counter in range((length + 31) // 32):
        blocks.append(
            hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        )
    return b"".join(blocks)[:length]


def _xor(data: bytes, stream: bytes) -> bytes:
    return (int.from_bytes(data, "big") ^ int.from_bytes(stream, "big")).to_bytes(
        len(data), "big"
    )


def encrypt(plaintext: bytes, secret: str) -> bytes:
    enc_key, mac_key = _keys(secret)
    nonce = os.urandom(16)
    ciphertext = _xor(plaintext, _keystream(enc_key, nonce, len(plaintext)))
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    return MAGIC + nonce + tag + ciphertext


def decrypt(blob: bytes, secret: str) -> bytes:
    if not blob.startswith(MAGIC):
        raise ReferenceError("not an epydemix reference archive")
    body = blob[len(MAGIC) :]
    nonce, tag, ciphertext = body[:16], body[16:48], body[48:]
    enc_key, mac_key = _keys(secret)
    expected = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise ReferenceError("wrong key, or the archive was modified")
    return _xor(ciphertext, _keystream(enc_key, nonce, len(ciphertext)))


def secret_from_env() -> str:
    secret = os.environ.get(KEY_ENV)
    if not secret:
        raise ReferenceError(
            f"set {KEY_ENV} to the benchmark key to use the references"
        )
    return secret


def pack(tasks_dir: Path, archive: Path, secret: str, private_tasks) -> list:
    """Encrypt ``<task>/reference/`` of every private task into ``archive``."""
    buffer = io.BytesIO()
    packed = []
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for task in sorted(private_tasks):
            ref = Path(tasks_dir) / task / "reference"
            if not ref.is_dir():
                raise ReferenceError(f"{task} has no reference/ to pack")
            for path in sorted(ref.rglob("*")):
                if path.is_file():
                    info = tar.gettarinfo(
                        str(path),
                        arcname=f"{task}/reference/{path.relative_to(ref).as_posix()}",
                    )
                    info.mtime, info.uid, info.gid, info.uname, info.gname = (
                        0,
                        0,
                        0,
                        "",
                        "",
                    )
                    with open(path, "rb") as f:
                        tar.addfile(info, f)
            packed.append(task)
    Path(archive).write_bytes(encrypt(buffer.getvalue(), secret))
    return packed


def unpack(archive: Path, tasks_dir: Path, secret: str) -> list:
    """Decrypt the archive into ``<task>/reference/``; returns the tasks."""
    data = decrypt(Path(archive).read_bytes(), secret)
    root = Path(tasks_dir).resolve()
    tasks = set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            parts = Path(member.name).parts
            if (
                not member.isfile()
                or root not in target.parents
                or len(parts) < 3
                or parts[1] != "reference"
            ):
                raise ReferenceError(f"refusing to extract {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src:
                target.write_bytes(src.read())
            tasks.add(parts[0])
    return sorted(tasks)
