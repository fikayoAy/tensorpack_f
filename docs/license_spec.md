License JSON spec and vendor signing checklist

Purpose

This document defines a canonical JSON license format, the canonicalization rules for signing, recommended signing algorithms, and a short vendor checklist for issuing signed licenses the app can verify offline.

1. Canonical license JSON schema (recommended fields)

- license_id: string (unique, e.g. UUID or JTI)
- product_id: string (e.g. "tensorpack-premium")
- customer_id: string (email or account id)
- license_type: string ("free", "trial", "academic", "premium")
- issued_at: string (ISO8601 UTC)
- expires_at: string or null (ISO8601 UTC)
- machine_fingerprint: string or null (vendor-bound fingerprint/hash)
- features: object (map of entitlement keys -> values)
- offline_grace_days: integer (optional, days allowed offline after expiry)
- version: integer (schema version)
- meta: object (optional vendor metadata)
- jti: string (unique token id for revocation lookups)

Wrapper for signature:
- payload: the canonical JSON object (above)
- signature: base64-encoded signature bytes
- sig_alg: string (e.g. "RS256" or "Ed25519")
- key_id: optional string (public key identifier)

2. Canonicalization rules (must be identical for signer and verifier)

- Encode payload as UTF-8.
- Serialize the payload JSON with keys sorted (lexicographic).
- Use no extra whitespace (most JSON serializers with sort_keys=True produce deterministic output).
- Use ISO8601 UTC for datetimes with trailing Z (e.g. 2025-09-09T17:00:00Z).
- Numbers should be plain JSON numbers (avoid locale formatting).
- Do not include transient fields (e.g. last-verification timestamps) inside signed payload.

3. Recommended signing algorithms

- Ed25519: compact, fast, and safe; preferred if available.
- RSA-PSS with SHA-256 (RSASSA-PSS): widely supported; use key sizes >= 2048.
- Provide key_id in the wrapper to allow key rotation.

4. Vendor signing checklist

- Generate key pair offline and protect the private key.
  - Ed25519: use libsodium or openssh-keygen workflows.
  - RSA: use openssl to generate RSA key (2048/3072+).
- Compose the payload JSON following the schema.
- Serialize with canonical rules (sorted keys, UTF-8).
- Sign the serialized bytes with the vendor private key.
- Produce a license file containing {"payload": <object>, "signature": "<b64>", "sig_alg": "Ed25519", "key_id": "v1"}
- Deliver the license file to the customer by secure channel.
- Record jti and customer mapping in vendor DB for revocation.

5. App verification checklist (what the app must do)

- Load license file and extract payload, signature, sig_alg, key_id.
- Re-serialize payload using the canonical rules and verify signature with the public key matching key_id.
- Check payload.product_id matches expected product.
- Check time validity: issued_at <= now <= expires_at (or within offline_grace_days if offline policy allows).
- If machine_fingerprint present, verify match (allow documented fallback behavior).
- Check jti against revocation store (if online) or keep a short-lived revocation cache updated periodically.
- On success, cache the verified license metadata and last-verified timestamp.

6. Revocation and refresh policy

- Issue short expiries (e.g. 7 days) for offline tokens and require online refresh periodically.
- Keep a server-side revocation list keyed by jti; the app should refresh revocation cache on periodic online checks.
- Allow an offline grace (e.g. 48–72 hours) after expiry to accommodate disconnected use; after grace expires, downgrade to free.

7. Machine fingerprint suggestions

- Prefer hardware-backed attestation (TPM) when available.
- Fallback: hash of (hostname + primary MAC + stable machine-id); document how to compute so signer and verifier agree.
- Avoid fragile identifiers like IP addresses.

8. File layout and example (illustrative)

{
  "payload": {
    "license_id": "b4f6d1a2-...",
    "product_id": "tensorpack-premium",
    "customer_id": "alice@example.com",
    "license_type": "premium",
    "issued_at": "2025-09-01T12:00:00Z",
    "expires_at": "2025-09-08T12:00:00Z",
    "machine_fingerprint": "sha256:...",
    "features": {"max_datasets": "inf"},
    "offline_grace_days": 2,
    "version": 1,
    "jti": "jti-12345"
  },
  "signature": "<base64-signature>",
  "sig_alg": "Ed25519",
  "key_id": "v1"
}

9. Key rotation

- Support multiple public keys in the app; key_id lets the app pick the correct public key.
- When rotating, sign new licenses with new key_id and keep old public keys for verification until all active licenses expire.

10. Minimal vendor API (optional)

- POST /activate {license_key} -> returns signed license payload or error.
- GET /revoked?jti=... -> returns revoked boolean.
- Provide TLS and authenticated access for vendor portal.

Next steps I can take for you

- Implement the app-side verification changes in `license_manager.py` so local license files are accepted only when signature verifies (requires `cryptography` or equivalent in container).
- Produce a tiny signing script for the vendor (example commands) to create signed files.
- Create a minimal mock activation endpoint (Flask) for local testing.

Tell me which one to do next and I'll implement it.
