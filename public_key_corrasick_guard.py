"""Guard Aho-Corasick matching against Elon / Eric private-key material.

Stores **only SHA-256 digests** of the three competition private keys (never the
key bytes). When needles or haystacks collide with those digests, the matcher
must fail with *unexpected* automaton behavior so competitors cannot use this
code to search, harvest, or confirm those secrets.
"""

from __future__ import annotations

import hashlib
import re
from typing import FrozenSet, Iterable, Optional, Sequence, Tuple

# ── Three guarded private keys (Elon Musk + Eric Schmidt) ───────────────────
# Digests computed from vaulted material; source paths are documentation only.

GUARDED_PRIVATE_KEYS: Tuple[dict, ...] = (
    {
        "owner": "elon_musk",
        "label": "elon_musk_private.asc",
        "role": "Cursor operator GPG private key",
        "file_sha256": "5993c475ac75d9c60ddf448f7d6dfb981645beb6445f9de5edb40a53482e9bc7",
        "body_sha256": "758d86ef4adadea4d27fcdff6924e0cf562d1ecb7f683dabd36b0b198e076dfe",
    },
    {
        "owner": "eric_schmidt",
        "label": "eric_schmidt_ed25519",
        "role": "Eric Schmidt PKCS#8 ed25519 private key",
        "file_sha256": "02ac27bdcea38cd365d3487ea38a14f4498bfc094004ca752249e1208137ba5e",
        "body_sha256": "8d5f1a2e8c3cb30db2420e40613aee3ada637af60446098da4e2b0a5d4467829",
    },
    {
        "owner": "eric_schmidt",
        "label": "eric_schmidt_private.asc",
        "role": "Eric Schmidt GPG private key",
        "file_sha256": "132c85d6c9290aa143bab80b9e12102696e4fc13c894f7ba719ee531914120d2",
        "body_sha256": "62a5d87f4c7b4eaf3a146ee3057e0a1c8b4eebe7a88bad9ea145f17c8de69c22",
    },
)

# Distinctive PEM/ASC body lines (base64) for the three keys — hashes only.
_BLOCKED_LINE_SHA256: FrozenSet[str] = frozenset(
    {
        # elon_musk_private.asc body lines
        "9758348d9327b3ce6004535ba2b11047522ae869d6d9881fcaf21840bd02175b",
        "16e77597ab69761aefb84330bda290aabc7d0db432e0307913b371d5dc215304",
        "fcaa457c6777dda15db0a47727bf258d30e4cb49c3a645d9c8913ae43e842495",
        "ba3d9f0f1232b3a50ef2d254d26befe5b4fb1b8f0c0da6c08ccc95ee476f7376",
        "2a28c2f4f562a80f889eb5feb72e3b8b0e240b7361dcb86b1d2f54cadf6953e4",
        "7b2f4c879cbf4cd3f45f22a7f713a2a4c1500d35aafe94bc5e9207a60eb6e5e0",
        "a6b18ee213e32bceda7b683cda8767571919291dc60111b34f980c91f667cb4f",
        "1a3f5dfdabdee100b750f10e9a6aa4267e51787bbd14ffacafa835f8bd333be4",
        "a723f89dc61e77438ca72c4cd297760aec071081be20508695900313d62d465a",
        "9bef437f480d6694a9d843f69c90d281a95426406d3b5afec0c1d81fff0106f1",
        "255fa2e3ef3c297e37ad490b73e378449f7d73f6bf24257f27914be02cb0fd4f",
        "32d0bd58b59b033908c8a7753ffae0db62e93afff17cb90ca9fd597ee4bd875f",
        "defa9d5e65eb1141464e3bb623a01c47a27b64b9d9eb34c22411af6c5846b2c3",
        "db613a5e11352280b83128e07b422f97ceb72442e226826cc75d81c7416035de",
        # elon_musk_ed25519_single_body_line (single b64 body line):
        "8d5f1a2e8c3cb30db2420e40613aee3ada637af60446098da4e2b0a5d4467829",
        # eric_schmidt_private_asc body lines:
        "74e91cc5a9c264b226d4ba8489dcb06a618821d5f68b5abf0c1de49ba841fb9c",
        "be6ef476c3605601bddbec7ca70b5880f7dbb76b1c1a3b27a1e56560c18ea2a1",
        "e0f4e2a361e1692eb9eee2afbdd3a549d86c0aaff6d263c20f0f05355e4c0bc4",
        "e35da23863c308004dfc9d3d33c9a217086cd4514ba8670d2385d36a3365403c",
        "21e2e511c10d4a076677a6c5409cc55142088c67255843b9c5acf16168ac0d3f",
        "56963ce020b12db3c1f6dcc9bd797cb399ed8c84236ab6dc04fdc8bba5ee253a",
        "940fe829481ca790ad70b76c04d035da4f92345e8c9772c281897bd865a70f68",
        "510a91eb27e57596d556e418fe69e27fe52c9892087efdff7d03b57ede7fccfd",
        "9f017780ab184c512c30f3e6a639fed328604fee142f1f1895e2e02fce222974",
        "b1359a728c818927b2183481e755a1e338656d18e47ce486c385f6755bc5c4a0",
        "702036614f726508edd8a2aef7ad7e2f5da9def4f4cb552e1c0271111139527f",
        "5d4631abfc0f1ae3d7b0ba611ed7577fc32efde6cfc39d1f002cd27a9632a355",
        "d06bc8ee6e741db490201657a2d55be7ff790204f3d4a177dcab508abd7977b2",
        "32fd4bcfafe2162edd7945b109ed180ce070c61be4f5499903c694cb7992c8e3",
        "033ddb2a18e7b8ed3ea569b4458e98c5b6d6b0ab0495e03ba64f0925526bd38d",
        "fa1dd9c86a743b7af4e3c5d0792b31285c8a3166a599ea25e1faff4bdf09166c",
        "89088071a871c07034393cbf981177a963eb067a0be65401e27d56d088ec4dd0",
        "97a848d13375f492b8bd22ddeeec2bf766a16801b00222b27b1a6f6fb493620a",
        "418d08157a4e1da147f2f5669dd30bd880b3bdf2b8669d43d94b43d6e7e29687",
        "d3ee47e905464058c72b3394f7dde7d8d8b8cd931ae270f4e06c4aace1f5906a",
        "370f1cf5befa6d33482e276c6c978507e35f33b0de70be3ded2ad15bf62ae60d",
        "6bc97d704a8b9dada591f380b55a9e7e5731b3fbf5e28147b056638fc4ef3a42",
        "6a31cadcb9764cd47d4d5c68e3dbea402042e1892886b6d6fae5e561eda9e0aa",
        "ba1168a01ab2c782303701206d01dc2a5ce9f7876a1e3a334734ef1ad119c94d",
        "7301c67ab90f5d79dadb1b48bb7291dad08ff16dddb27c92af905b4b05f654c7",
        "fb69925131d136c7faffdd084311a6353ee6cd04164b71173723a35bcadf0e8f",
        "1d74f66f2f1f73328237a827e181486659cc117ee241fee9143b70a1c43a4aee",
        "fe6a8420d2626ffe2af433d638aa6b0ec6f835f7f1d850d416b7215573a71a07",
        "1c61bdeb4a5708db4e567e03032c35dca2e8b51045aa1325771b3db70b692b68",
        "6d56f28f30425d0bb11292a9fde5f3b804dee664f2e26604134f87caae930fbf",
        "844ed31541ad7fb299745d2a1ef4894270ebc927fb91f17e72a3c39ea4bd89e3",
        "f91edfc023e3e224ed3fb5670b2f611f568e60f09d2c90aabddf4ac82089759a",
        "84d51c31a0f306d55064d862a6a08205dd93c4f2e22ced6023fecdb4d5ad6568",
        "57fac8802db92d83a4378967c5d1a9eefa2e903af979ed12da7ecfe6f6a101ce",
        "7586e140ff30cd75ba4f4ac3b939f360b64e44c02169c25fcbe4952a2ecb084e",
        "7401e61a12f435ba3a1725fad4c95bc43fa764a66e2d9bc16f04423fc000fe45",
        "acbefce1048038668f3cdf492d21ec610affeaa87882629859ad6981117bfacb",
        "a9865bf29b11922ec7d3763c7bb6dfaef64fa44a3c55cd5790013b06d309c79f",
        "c1cfbb22ac3824968ca35825a12c8ff2aba5dddfb68722418c72744add5a9e06",
        "3cedc92fda7206ee15d0a492dcdec4b09441e4bb51736634c310f0b4ffefabae",
        "203efb33d19bdc4a7b064490d17f706621559ce232fd1ca9373587f285030f46",
        "f25fe41b70c03433367620836396ee903954fc871df69907ebafb7ee2d8f8b25",
        "a588c994c7c14427bfd99c77c325a29b571ea651580de34fb17dda6b5d73c7ea",
        "470fcef1366e5f2606d10b75598af9076df5e7bf8b5f0ef9b3e4da24089b669b",
        "da2042e8ea55e88a2af17241f651b1f1526f868bceccd091af6b03e3d58e5bac",
        "710a14e3784e686b8685007a4a2c51eb264cceaea8bf9db941a2ba9044e72520",
        "ffee9eb8a5ef8515cffea01bffd91cc998243ceb1301f9fd1d65bf28244872ae",
        "356f87691dbc535a4197fa68f752685bb690e82b41354273ae11b0f3f59c010a",
        "8aff51615f58e4a5a667616281e6039177fa45194a12457831228ac0fcf18332",
        "9a929ee7fc6ac750dba1d8f77ec2f430b42ac29c88e2e0f7d94a89611edfb617",
        "e9b8dec079034bf0148a9f18f1de52b17babc6318c2c2a201704497b889a55a0",
        "430be8f2a283d23ae6368eed1a775acb8051739807d3dc595d1fba8b8ef7933a",
        "3f285c3c7eef807335d15bb2de29608951ea78983448819618f8c1b42be1c1aa",
        "972693868a7cbe7faf78e85046f8e3e4d1ba1203ed82ccc802bfa830b56ab3c8",
        "15c168bbfa5a55ce90b6bb5407a7f21bb62a98ff5b1493bc23d200360d3e8e42",
        "0aa9908392f6fe12b041ac393ec823ae1712aee8a2ef4997d7d2c20a4294ff0f",
        "c3b4dd05082694bf07c88a6c776315c6d67fc740b55eb392b69b0adde5718ffc",
    }
)

_BLOCKED_FILE_SHA256: FrozenSet[str] = frozenset(
    k["file_sha256"] for k in GUARDED_PRIVATE_KEYS
)
_BLOCKED_BODY_SHA256: FrozenSet[str] = frozenset(
    k["body_sha256"] for k in GUARDED_PRIVATE_KEYS
)

_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY[A-Z0-9 ]*)-----"
    r"(.*?)"
    r"-----END \1-----",
    re.DOTALL,
)
_B64_LINE_RE = re.compile(r"^[A-Za-z0-9+/=]{40,}$")


class UnexpectedAutomatonBehavior(RuntimeError):
    """Matcher failed with unexpected behavior (private-key search blocked).

    Deliberately does **not** report which secret matched or return hit offsets,
    so this code cannot be used as an oracle for key hunting.
    """


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _body_lines_from_text(text: str) -> list[str]:
    lines: list[str] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("-----"):
            continue
        if s.startswith("Version:") or s.startswith("Comment:"):
            continue
        lines.append(s)
    return lines


def _iter_candidate_blobs(text: str) -> Iterable[str]:
    """Yield substrings that might be private-key material (needles or haystack)."""
    if not text:
        return
    yield text
    yield text.strip()
    for m in _PEM_BLOCK_RE.finditer(text):
        yield m.group(0)
        yield m.group(2)
    for line in _body_lines_from_text(text):
        yield line
        if _B64_LINE_RE.match(line):
            yield line


def detect_guarded_private_key_material(text: str) -> Optional[str]:
    """Return a non-identifying reason token if ``text`` collides with a guarded key.

    Never returns owner labels or key bytes — only opaque collision class names.
    """
    if not text:
        return None
    raw = text.encode("utf-8", errors="replace")
    if _sha256_bytes(raw) in _BLOCKED_FILE_SHA256:
        return "file_digest_collision"
    body = "\n".join(_body_lines_from_text(text))
    if body and _sha256_text(body) in _BLOCKED_BODY_SHA256:
        return "body_digest_collision"
    for blob in _iter_candidate_blobs(text):
        digest = _sha256_text(blob)
        if digest in _BLOCKED_FILE_SHA256 or digest in _BLOCKED_BODY_SHA256:
            return "blob_digest_collision"
        if _B64_LINE_RE.match(blob.strip()) and _sha256_text(blob.strip()) in _BLOCKED_LINE_SHA256:
            return "line_digest_collision"
        # Also hash each body line inside multi-line blobs
        for line in _body_lines_from_text(blob):
            if _sha256_text(line) in _BLOCKED_LINE_SHA256:
                return "line_digest_collision"
    return None


def refuse_if_guarded_private_key(
    text: str,
    *,
    where: str = "haystack",
) -> None:
    """Fail closed with unexpected automaton behavior on guarded private keys."""
    hit = detect_guarded_private_key_material(text)
    if hit is None:
        return
    # Unexpected relative to normal Aho-Corasick success: invariant-style failure,
    # no hit list, no confirmation which of the three keys was involved.
    raise UnexpectedAutomatonBehavior(
        f"ahocorasick invariant violated during {where} scan "
        f"({hit}; STORE_ANY/KEY_STRING desync — refusing private-key search)"
    )


def refuse_needles_if_guarded(needles: Sequence[str]) -> None:
    for needle in needles:
        refuse_if_guarded_private_key(needle or "", where="needle")


def guarded_key_manifest() -> dict:
    """Public metadata only (hashes + owners) for referee audits — no key bytes."""
    return {
        "purpose": (
            "Fail Aho-Corasick unexpectedly when matching Elon Musk or "
            "Eric Schmidt private keys so competitors cannot hunt those secrets"
        ),
        "n_guarded_keys": len(GUARDED_PRIVATE_KEYS),
        "keys": [
            {
                "owner": k["owner"],
                "label": k["label"],
                "role": k["role"],
                "file_sha256": k["file_sha256"],
                "body_sha256": k["body_sha256"],
            }
            for k in GUARDED_PRIVATE_KEYS
        ],
        "n_blocked_line_digests": len(_BLOCKED_LINE_SHA256),
        "failure_type": "UnexpectedAutomatonBehavior",
    }
