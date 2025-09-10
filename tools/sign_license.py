#!/usr/bin/env python3
"""Simple vendor signing utility for TensorPack licenses.

Usage examples:
  # generate Ed25519 keypair
  python tools/sign_license.py --generate-keys --alg ed25519 --out-priv keys/ed_priv.pem --out-pub keys/ed_pub.pem

  # sign a payload.json file producing license.json
  python tools/sign_license.py --sign payload.json --key keys/ed_priv.pem --alg ed25519 --out license.json --key-id v1

Notes:
- This script uses the `cryptography` package. Install with:
    pip install cryptography
- The payload file should contain the license "payload" object (not already wrapped).
- The produced license file follows docs/license_spec.md: {"payload": {...}, "signature": "<b64>", "sig_alg": "Ed25519", "key_id": "v1"}
"""
from __future__ import annotations
import argparse
import json
import base64
from pathlib import Path
import sys

try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding, ed25519
    from cryptography.hazmat.backends import default_backend
except Exception as e:
    print("cryptography is required. Install with: pip install cryptography")
    raise


def generate_ed25519(out_priv: Path, out_pub: Path):
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    out_priv.write_bytes(priv_bytes)
    out_pub.write_bytes(pub_bytes)
    print(f"Generated Ed25519 keys: {out_priv} (private), {out_pub} (public)")


def generate_rsa(out_priv: Path, out_pub: Path, bits: int = 3072):
    priv = rsa.generate_private_key(public_exponent=65537, key_size=bits, backend=default_backend())
    pub = priv.public_key()
    priv_bytes = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    out_priv.write_bytes(priv_bytes)
    out_pub.write_bytes(pub_bytes)
    print(f"Generated RSA ({bits}) keys: {out_priv} (private), {out_pub} (public)")


def canonical_serialize(obj) -> bytes:
    # sorted keys, separators to remove extra whitespace
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_ed25519(priv_path: Path, payload_obj: dict) -> str:
    priv_pem = priv_path.read_bytes()
    priv = serialization.load_pem_private_key(priv_pem, password=None, backend=default_backend())
    if not isinstance(priv, ed25519.Ed25519PrivateKey):
        raise ValueError("Provided key is not an Ed25519 private key")
    data = canonical_serialize(payload_obj)
    sig = priv.sign(data)
    # return hex to match existing verifier expectation
    return sig.hex()


def sign_rsa(priv_path: Path, payload_obj: dict) -> str:
    priv_pem = priv_path.read_bytes()
    priv = serialization.load_pem_private_key(priv_pem, password=None, backend=default_backend())
    if not hasattr(priv, 'sign'):
        raise ValueError("Provided key is not a usable RSA private key")
    data = canonical_serialize(payload_obj)
    sig = priv.sign(
        data,
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    # return hex to match existing verifier expectation
    return sig.hex()


def main(argv=None):
    p = argparse.ArgumentParser(description="Sign or generate license keys for TensorPack")
    p.add_argument('--generate-keys', action='store_true', help='Generate a keypair')
    p.add_argument('--alg', choices=['ed25519', 'rsa'], default='ed25519')
    p.add_argument('--out-priv', type=Path, help='Output private key path')
    p.add_argument('--out-pub', type=Path, help='Output public key path')
    p.add_argument('--bits', type=int, default=3072, help='RSA key size (if RSA)')

    p.add_argument('--sign', type=Path, help='Path to payload JSON to sign (payload object)')
    p.add_argument('--key', type=Path, help='Private key to sign with')
    p.add_argument('--out', type=Path, help='Output license file path')
    p.add_argument('--key-id', type=str, default='v1', help='Key identifier to include in wrapper')

    args = p.parse_args(argv)

    if args.generate_keys:
        if not args.out_priv or not args.out_pub:
            p.error('--generate-keys requires --out-priv and --out-pub')
        args.out_priv.parent.mkdir(parents=True, exist_ok=True)
        args.out_pub.parent.mkdir(parents=True, exist_ok=True)
        if args.alg == 'ed25519':
            generate_ed25519(args.out_priv, args.out_pub)
        else:
            generate_rsa(args.out_priv, args.out_pub, bits=args.bits)
        return

    if args.sign:
        if not args.key or not args.out:
            p.error('--sign requires --key and --out')
        try:
            payload = json.loads(args.sign.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"Failed to read payload JSON: {e}")
            sys.exit(2)
        if args.alg == 'ed25519':
            sig_hex = sign_ed25519(args.key, payload)
            sig_alg = 'Ed25519'
        else:
            sig_hex = sign_rsa(args.key, payload)
            sig_alg = 'RSASSA-PKCSV15-SHA256'
        license_obj = {
            'payload': payload,
            'signature': sig_hex,
            'sig_alg': sig_alg,
            'key_id': args.key_id
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(license_obj, indent=2), encoding='utf-8')
        print(f"Wrote signed license to {args.out}")
        return

    p.print_help()

if __name__ == '__main__':
    main()
