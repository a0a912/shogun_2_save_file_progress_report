#!/usr/bin/env python3
"""
shogun2_report.py

Read-only Total War: SHOGUN 2 campaign-save report generator.

What it extracts from CAAB-format Shogun 2 .save files:
  * Internal faction, turn, year, and season (ignores misleading filenames)
  * Current treasury
  * Persistent faction EVENT_LOG / major campaign history
  * Battle record and opponent/result breakdowns
  * Province capture history
  * Diplomacy, character, development, and major-clan milestones
  * Rolling FACTION_ECONOMICS history (raw arrays, because their field labels
    are not fully documented here)
  * A few current-state record counts (army and character records)

Outputs:
  * A self-contained HTML report
  * A JSON file containing the extracted structured data

The script NEVER writes to the .save file.

Usage:
    python3 shogun2_report.py "Quick Save.save"
    python3 shogun2_report.py save1.save save2.save --out-dir reports

No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import html
import json
import lzma
import math
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


# ---------------------------------------------------------------------------
# Minimal CAAB ESF reader
# ---------------------------------------------------------------------------


class ESFError(RuntimeError):
    pass


@dataclass(slots=True)
class Node:
    kind: str
    value: Any = None
    name: Optional[str] = None
    version: Optional[int] = None
    flags: Optional[int] = None
    groups: list[list["Node"]] = field(default_factory=list)


class Reader:
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def tell(self) -> int:
        return self.pos

    def seek(self, pos: int) -> None:
        if not (0 <= pos <= len(self.data)):
            raise ESFError(f"Invalid seek to {pos}")
        self.pos = pos

    def read(self, size: int) -> bytes:
        end = self.pos + size
        if end > len(self.data):
            raise ESFError(
                f"Unexpected end of data at 0x{self.pos:x}; wanted {size} bytes"
            )
        out = self.data[self.pos:end]
        self.pos = end
        return out

    def u8(self) -> int:
        return self.read(1)[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.read(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.read(2))[0]

    def u24(self) -> int:
        b = self.read(3)
        return b[0] | (b[1] << 8) | (b[2] << 16)

    def i24(self) -> int:
        value = self.u24()
        return value - (1 << 24) if value & (1 << 23) else value

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    def cauleb128(self) -> int:
        """Creative Assembly's big-endian-ish ULEB128 variant."""
        value = 0
        while True:
            byte = self.u8()
            value = (value << 7) | (byte & 0x7F)
            if not (byte & 0x80):
                return value

    def sized_u8_string(self) -> str:
        size = self.u16()
        return self.read(size).decode("utf-8", errors="replace")

    def sized_u16_string(self) -> str:
        chars = self.u16()
        return self.read(chars * 2).decode("utf-16le", errors="replace")


class CAABESF:
    """Read the CAAB ESF variant used by the supplied Shogun 2 saves."""

    __slots__ = (
        "data",
        "signature",
        "unknown_1",
        "creation_date",
        "record_names",
        "strings_utf8",
        "strings_utf16",
        "root",
        "padding_bytes",
    )

    def __init__(self, data: bytes):
        self.data = data
        self.signature = ""
        self.unknown_1 = 0
        self.creation_date = 0
        self.record_names: list[str] = []
        self.strings_utf8: dict[int, str] = {}
        self.strings_utf16: dict[int, str] = {}
        self.root: Node
        self.padding_bytes = 0
        self._parse()

    def _parse(self) -> None:
        r = Reader(self.data)
        signature = r.read(4)
        if signature != b"\xCA\xAB\x00\x00":
            raise ESFError(
                "Unsupported ESF signature. This version currently supports CAAB "
                "Shogun 2 saves only."
            )

        self.signature = "CAAB"
        self.unknown_1 = r.u32()
        self.creation_date = r.u32()
        string_table_offset = r.u32()
        node_offset = r.tell()

        if not (node_offset <= string_table_offset <= len(self.data)):
            raise ESFError("Invalid CAAB string-table offset")

        # Read string tables first because the node tree references them by index.
        r.seek(string_table_offset)
        record_count = r.u16()
        self.record_names = [r.sized_u8_string() for _ in range(record_count)]

        utf16_count = r.u32()
        for _ in range(utf16_count):
            value = r.sized_u16_string()
            index = r.u32()
            self.strings_utf16[index] = value

        utf8_count = r.u32()
        for _ in range(utf8_count):
            value = r.sized_u8_string()
            index = r.u32()
            self.strings_utf8[index] = value

        # The supplied Shogun 2 saves contain a small all-zero trailer after the
        # string tables. Treat zero-only bytes as harmless padding, but reject
        # unparsed non-zero data.
        if r.tell() != len(self.data):
            trailer = self.data[r.tell() :]
            if any(trailer):
                raise ESFError(
                    f"Unexpected non-zero trailer after string tables "
                    f"({len(trailer)} bytes)"
                )
            self.padding_bytes = len(trailer)

        r.seek(node_offset)
        self.root = self._read_node(r, is_root=True)
        if r.tell() != string_table_offset:
            raise ESFError(
                f"Node-tree size mismatch: stopped at 0x{r.tell():x}, "
                f"expected 0x{string_table_offset:x}"
            )

    def _read_node(self, r: Reader, is_root: bool = False) -> Node:
        start = r.tell()
        marker = r.u8()

        # Record nodes: bit 7 set.
        if marker & 0x80:
            flags = marker
            if (flags & 0x20) or is_root:
                name_index = r.u16()
                version = r.u8()
            else:
                version = (marker & 0x1E) >> 1
                name_index = ((marker & 0x01) << 8) + r.u8()

            try:
                name = self.record_names[name_index]
            except IndexError as exc:
                raise ESFError(
                    f"Unknown record-name index {name_index} at 0x{start:x}"
                ) from exc

            block_size = r.cauleb128()
            group_count = r.cauleb128() if (flags & 0x40) else 1
            block_end = r.tell() + block_size
            groups: list[list[Node]] = []

            for _ in range(group_count):
                if flags & 0x40:
                    entry_size = r.cauleb128()
                    group_end = r.tell() + entry_size
                else:
                    group_end = block_end

                group: list[Node] = []
                while r.tell() < group_end:
                    group.append(self._read_node(r, is_root=False))

                if r.tell() != group_end:
                    raise ESFError(
                        f"Group-size mismatch in {name}: {r.tell()} != {group_end}"
                    )
                groups.append(group)

            if r.tell() != block_end:
                raise ESFError(
                    f"Record-size mismatch in {name}: {r.tell()} != {block_end}"
                )

            return Node(
                "record", name=name, version=version, flags=flags, groups=groups
            )

        # Standard primitives.
        if marker == 0x01:
            value = r.u8()
            if value not in (0, 1):
                raise ESFError(f"Invalid bool {value} at 0x{start:x}")
            return Node("bool", bool(value))
        if marker == 0x02:
            return Node("i8", r.i8())
        if marker == 0x03:
            return Node("i16", r.i16())
        if marker == 0x04:
            return Node("i32", r.i32())
        if marker == 0x05:
            return Node("i64", r.i64())
        if marker == 0x06:
            return Node("u8", r.u8())
        if marker == 0x07:
            return Node("u16", r.u16())
        if marker == 0x08:
            return Node("u32", r.u32())
        if marker == 0x09:
            return Node("u64", r.u64())
        if marker == 0x0A:
            return Node("f32", r.f32())
        if marker == 0x0B:
            return Node("f64", r.f64())
        if marker == 0x0C:
            return Node("coord2d", (r.f32(), r.f32()))
        if marker == 0x0D:
            return Node("coord3d", (r.f32(), r.f32(), r.f32()))
        if marker == 0x0E:
            index = r.u32()
            try:
                return Node("utf16", self.strings_utf16[index])
            except KeyError as exc:
                raise ESFError(f"Missing UTF-16 string index {index}") from exc
        if marker == 0x0F:
            index = r.u32()
            try:
                return Node("ascii", self.strings_utf8[index])
            except KeyError as exc:
                raise ESFError(f"Missing UTF-8 string index {index}") from exc
        if marker == 0x10:
            return Node("angle", r.i16())

        # Optimized primitives.
        if marker == 0x12:
            return Node("bool", True)
        if marker == 0x13:
            return Node("bool", False)
        if marker == 0x14:
            return Node("u32", 0)
        if marker == 0x15:
            return Node("u32", 1)
        if marker == 0x16:
            return Node("u32", r.u8())
        if marker == 0x17:
            return Node("u32", r.u16())
        if marker == 0x18:
            return Node("u32", r.u24())
        if marker == 0x19:
            return Node("i32", 0)
        if marker == 0x1A:
            return Node("i32", r.i8())
        if marker == 0x1B:
            return Node("i32", r.i16())
        if marker == 0x1C:
            return Node("i32", r.i24())
        if marker == 0x1D:
            return Node("f32", 0.0)

        # Known-but-undocumented types preserved as numeric/raw values.
        if marker == 0x21:
            return Node("unknown21", r.u32())
        if marker == 0x23:
            return Node("unknown23", r.u8())
        if marker == 0x24:
            return Node("unknown24", r.u16())
        if marker == 0x25:
            return Node("unknown25", r.u32())
        if marker == 0x26:
            first = r.u8()
            payload = bytes([first])
            if first % 8 == 0 and first != 0:
                payload += r.read(first)
            else:
                payload += r.read(7)
            # RPFM conditionally consumes a trailing 0x9C marker.
            if r.tell() < len(r.data):
                last = r.u8()
                if last != 0x9C:
                    r.pos -= 1
            return Node("unknown26", payload.hex())

        # Arrays. Their CAULEB128 prefix is a BYTE size, not an element count.
        if marker in {
            0x41,
            0x42,
            0x43,
            0x44,
            0x45,
            0x46,
            0x47,
            0x48,
            0x49,
            0x4A,
            0x4B,
            0x4C,
            0x4D,
            0x4E,
            0x4F,
            0x50,
            0x56,
            0x57,
            0x58,
            0x5A,
            0x5B,
            0x5C,
        }:
            byte_size = r.cauleb128()
            end = r.tell() + byte_size
            values: list[Any] = []

            while r.tell() < end:
                if marker == 0x41:
                    b = r.u8()
                    if b not in (0, 1):
                        raise ESFError("Invalid bool-array element")
                    values.append(bool(b))
                elif marker == 0x42:
                    # RPFM preserves this as raw u8 values.
                    values.append(r.u8())
                elif marker == 0x43:
                    values.append(r.i16())
                elif marker == 0x44:
                    values.append(r.i32())
                elif marker == 0x45:
                    values.append(r.i64())
                elif marker == 0x46:
                    values.extend(r.read(end - r.tell()))
                    break
                elif marker == 0x47:
                    values.append(r.u16())
                elif marker == 0x48:
                    values.append(r.u32())
                elif marker == 0x49:
                    values.append(r.u64())
                elif marker == 0x4A:
                    values.append(r.f32())
                elif marker == 0x4B:
                    values.append(r.f64())
                elif marker == 0x4C:
                    values.append((r.f32(), r.f32()))
                elif marker == 0x4D:
                    values.append((r.f32(), r.f32(), r.f32()))
                elif marker == 0x4E:
                    index = r.u32()
                    values.append(self.strings_utf16.get(index, f"<utf16:{index}>"))
                elif marker == 0x4F:
                    index = r.u32()
                    values.append(self.strings_utf8.get(index, f"<ascii:{index}>"))
                elif marker == 0x50:
                    values.append(r.i16())
                elif marker == 0x56:
                    values.append(r.u8())
                elif marker == 0x57:
                    values.append(r.u16())
                elif marker == 0x58:
                    values.append(r.u24())
                elif marker == 0x5A:
                    values.append(r.i8())
                elif marker == 0x5B:
                    values.append(r.i16())
                elif marker == 0x5C:
                    values.append(r.i24())

            if r.tell() != end:
                raise ESFError(f"Array-size mismatch for marker 0x{marker:02x}")

            kind_map = {
                0x41: "bool[]",
                0x42: "i8[]",
                0x43: "i16[]",
                0x44: "i32[]",
                0x45: "i64[]",
                0x46: "u8[]",
                0x47: "u16[]",
                0x48: "u32[]",
                0x49: "u64[]",
                0x4A: "f32[]",
                0x4B: "f64[]",
                0x4C: "coord2d[]",
                0x4D: "coord3d[]",
                0x4E: "utf16[]",
                0x4F: "ascii[]",
                0x50: "angle[]",
                0x56: "u32[]",
                0x57: "u32[]",
                0x58: "u32[]",
                0x5A: "i32[]",
                0x5B: "i32[]",
                0x5C: "i32[]",
            }
            value: Any = bytes(values) if marker == 0x46 else values
            return Node(kind_map[marker], value)

        raise ESFError(f"Unsupported ESF marker 0x{marker:02x} at 0x{start:x}")


def walk(node: Node) -> Iterator[Node]:
    yield node
    if node.kind == "record":
        for group in node.groups:
            for child in group:
                yield from walk(child)


def child_record(record: Node, name: str) -> Optional[Node]:
    if record.kind != "record" or not record.groups:
        return None
    for child in record.groups[0]:
        if child.kind == "record" and child.name == name:
            return child
    return None


def find_records(node: Node, name: str) -> Iterator[Node]:
    for candidate in walk(node):
        if candidate.kind == "record" and candidate.name == name:
            yield candidate


def _lzma_raw_decompress(properties: bytes, payload: bytes) -> bytes:
    """Decode the 5-byte LZMA1 properties used in Shogun 2's compressed node."""
    if len(properties) != 5:
        raise ESFError(f"Expected 5 LZMA property bytes; got {len(properties)}")

    prop = properties[0]
    lc = prop % 9
    rest = prop // 9
    lp = rest % 5
    pb = rest // 5
    dictionary_size = int.from_bytes(properties[1:5], "little")

    if lc > 8 or lp > 4 or pb > 4:
        raise ESFError("Invalid LZMA1 property byte")

    decoder = lzma.LZMADecompressor(
        format=lzma.FORMAT_RAW,
        filters=[
            {
                "id": lzma.FILTER_LZMA1,
                "dict_size": dictionary_size,
                "lc": lc,
                "lp": lp,
                "pb": pb,
            }
        ],
    )

    # These game streams do not necessarily carry an end-of-stream marker.
    # LZMADecompressor lets us use the decompressed bytes without requiring one.
    return decoder.decompress(payload)


def load_save(path: Path) -> tuple[CAABESF, dict[str, Any]]:
    outer = CAABESF(path.read_bytes())
    compression_info: dict[str, Any] = {"compressed": False}

    for node in walk(outer.root):
        if node.kind != "record" or node.name != "COMPRESSED_DATA" or not node.groups:
            continue
        group = node.groups[0]
        if len(group) < 2:
            continue
        payload_node, info_node = group[0], group[1]
        if (
            payload_node.kind != "u8[]"
            or info_node.kind != "record"
            or info_node.name != "COMPRESSED_DATA_INFO"
            or not info_node.groups
            or len(info_node.groups[0]) < 2
        ):
            continue

        declared = info_node.groups[0][0]
        properties_node = info_node.groups[0][1]
        if properties_node.kind != "u8[]":
            continue

        payload = payload_node.value
        properties = properties_node.value
        decoded = _lzma_raw_decompress(properties, payload)
        inner = CAABESF(decoded)
        compression_info = {
            "compressed": True,
            "compressed_payload_bytes": len(payload),
            "decompressed_bytes": len(decoded),
            "declared_compression_value": declared.value,
            "lzma_properties_hex": properties.hex(),
        }
        return inner, compression_info

    return outer, compression_info


# ---------------------------------------------------------------------------
# Shogun 2 campaign extraction
# ---------------------------------------------------------------------------


SEASONS = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}


def scalar_children(record: Node) -> list[Node]:
    if record.kind != "record" or not record.groups:
        return []
    return [x for x in record.groups[0] if x.kind != "record"]


def record_values(record: Node) -> list[Any]:
    if record.kind != "record" or not record.groups:
        return []
    return [x.value for x in record.groups[0]]


def parse_date_node(date_node: Optional[Node]) -> dict[str, Any]:
    if not date_node or date_node.kind != "record" or not date_node.groups:
        return {"year": None, "season_index": None, "season": "Unknown"}
    vals = [x.value for x in date_node.groups[0] if x.kind in {"u32", "i32"}]
    year = vals[0] if len(vals) > 0 else None
    season_index = vals[1] if len(vals) > 1 else None
    return {
        "year": year,
        "season_index": season_index,
        "season": SEASONS.get(season_index, f"Season {season_index}"),
        "raw": vals,
    }


def loc_strings(node: Optional[Node]) -> list[str]:
    if node is None:
        return []
    out: list[str] = []
    for candidate in walk(node):
        if candidate.kind == "record" and candidate.name == "CAMPAIGN_LOCALISATION":
            if candidate.groups and candidate.groups[0]:
                first = candidate.groups[0][0]
                if first.kind in {"utf16", "ascii"}:
                    out.append(str(first.value))
    return out


def _camel_words(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _smart_title(text: str) -> str:
    words = _camel_words(text).split()
    small = {"of", "the", "and", "to", "in", "for", "on"}
    out = []
    for i, word in enumerate(words):
        low = word.lower()
        if i and low in small:
            out.append(low)
        elif word.isupper() and len(word) <= 4:
            out.append(word)
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)


def humanize_loc(key: str) -> str:
    if not key:
        return ""

    known = {
        "random_localisation_strings_string_event_log_battle_victory_major_decisive": "Major/Decisive Victory",
        "random_localisation_strings_string_event_log_battle_victory_minor_close": "Minor/Close Victory",
        "random_localisation_strings_string_event_log_battle_victory_close_phyrric": "Close/Pyrrhic Victory",
        "random_localisation_strings_string_event_log_battle_defeat_major_decisive": "Major/Decisive Defeat",
        "random_localisation_strings_string_event_log_battle_defeat_minor_close": "Minor/Close Defeat",
        "random_localisation_strings_string_event_log_battle_defeat_close_phyrric": "Close/Pyrrhic Defeat",
        "random_localisation_strings_string_event_log_rebellion_christian": "Christian Rebels",
    }
    if key in known:
        return known[key]

    if key.startswith("factions_screen_name_"):
        return _smart_title(key.removeprefix("factions_screen_name_"))

    if key.startswith("regions_onscreen_jap_"):
        return _smart_title(key.removeprefix("regions_onscreen_jap_"))

    if key.startswith("names_name_names_"):
        value = key.removeprefix("names_name_names_")
        # These are name-pool prefixes, not necessarily part of the displayed name.
        for pool in ("shimazu", "japanese"):
            if value.startswith(pool) and len(value) > len(pool):
                value = value[len(pool) :]
                break
        return _smart_title(value)

    if key.startswith("random_localisation_strings_string_character_type_title_"):
        return _smart_title(
            key.removeprefix("random_localisation_strings_string_character_type_title_")
        )

    if key.startswith("technologies_onscreen_name_"):
        value = key.removeprefix("technologies_onscreen_name_")
        value = re.sub(r"^shogun_(civil|military)_(core|province)_", "", value)
        return _smart_title(value)

    if key.startswith("building_culture_variants_name_"):
        value = key.removeprefix("building_culture_variants_name_")
        value = re.sub(r"japanese$", "", value, flags=re.I)
        return _smart_title(value)

    if key.startswith("agent_culture_details_onscreen_name_"):
        number = key.rsplit("_", 1)[-1]
        return f"Character role #{number}"

    return _smart_title(key)


def event_category(event_type: str) -> str:
    if event_type == "battle_fought":
        return "Battles"
    if event_type in {"clan_region_taken", "rebel_outbreak", "rebel_crushed"}:
        return "Territory"
    if event_type.startswith("diplo_"):
        return "Diplomacy"
    if event_type.startswith("char_") or event_type.startswith("assassination_"):
        return "Characters"
    if event_type.startswith("clan_development_"):
        return "Development"
    if (
        event_type.startswith("clan_")
        or event_type.startswith("campaign_")
        or event_type.startswith("first_time_")
        or event_type.startswith("rel_")
    ):
        return "Milestones"
    return "Other"


def _name_pair(args: list[str], start: int = 0) -> str:
    bits = [humanize_loc(x) for x in args[start : start + 2]]
    return " ".join(x for x in bits if x).strip()


def describe_event(event_type: str, raw_args: list[str]) -> str:
    args = [humanize_loc(x) for x in raw_args]

    if event_type == "battle_fought":
        opponent = args[0] if args else "Unknown opponent"
        result = args[1] if len(args) > 1 else "Unknown result"
        return f"Battle against {opponent} — {result}"

    if event_type == "clan_region_taken":
        region = args[0] if args else "Unknown region"
        faction = args[1] if len(args) > 1 else "Unknown faction"
        return f"{region} taken by {faction}"

    if event_type == "clan_destroyed":
        return f"{args[0]} destroyed" if args else "Clan destroyed"

    if event_type == "clan_emerges":
        if len(args) >= 2:
            return f"{args[0]} emerges in {args[1]}"
        return "Clan emerges" + (f" — {', '.join(args)}" if args else "")

    if event_type == "diplo_war":
        return (
            f"War: {args[0]} ↔ {args[1]}"
            if len(args) >= 2
            else "War declared" + (f" — {', '.join(args)}" if args else "")
        )

    if event_type == "diplo_peace":
        return (
            f"Peace: {args[0]} ↔ {args[1]}"
            if len(args) >= 2
            else "Peace agreement" + (f" — {', '.join(args)}" if args else "")
        )

    if event_type == "diplo_alliance":
        return (
            f"Alliance: {args[0]} + {args[1]}"
            if len(args) >= 2
            else "Alliance formed" + (f" — {', '.join(args)}" if args else "")
        )

    if event_type == "diplo_alliance_broken":
        return (
            f"Alliance broken: {args[0]} / {args[1]}"
            if len(args) >= 2
            else "Alliance broken" + (f" — {', '.join(args)}" if args else "")
        )

    if event_type == "diplo_clan_encountered":
        return f"Encountered {args[0]}" if args else "Clan encountered"

    if event_type.startswith("char_dies_"):
        reason = event_type.removeprefix("char_dies_").replace("_", " ")
        if len(raw_args) >= 2:
            name = _name_pair(raw_args)
            extra = f" ({args[2]})" if len(args) > 2 else ""
            return f"{name} dies — {reason}{extra}"
        return f"Character dies — {reason}"

    if event_type == "char_succession":
        if len(raw_args) >= 4:
            new = _name_pair(raw_args, 0)
            old = _name_pair(raw_args, 2)
            return f"Succession: {new} succeeds {old}"
        return "Succession" + (f" — {', '.join(args)}" if args else "")

    if event_type == "assassination_successful":
        if len(raw_args) >= 2:
            name = _name_pair(raw_args)
            faction = args[2] if len(args) > 2 else ""
            return f"Successful assassination: {name}" + (f" ({faction})" if faction else "")
        return "Successful assassination"

    if event_type == "clan_development_civil":
        return f"Civil technology: {args[0]}" if args else "Civil technology completed"

    if event_type == "clan_realm_divide":
        return "Realm Divide"
    if event_type == "clan_fame_imperial_recognition":
        return "Imperial Recognition"
    if event_type == "clan_fame_national_renown":
        return "National Renown"
    if event_type == "clan_fame_shoguns_ire":
        return "Shogun's Ire"
    if event_type == "campaign_won":
        return "Campaign won"
    if event_type == "clan_player_becomes_shogun":
        name = _name_pair(raw_args) if len(raw_args) >= 2 else "Player daimyo"
        return f"{name} becomes Shogun"
    if event_type == "first_time_europeans":
        if len(args) >= 2:
            return f"Europeans arrive — {args[0]} / {args[1]}"
        return "Europeans arrive"
    if event_type == "first_time_legendary":
        return "Legendary building/event" + (f" — {', '.join(args)}" if args else "")
    if event_type == "rel_christianity_adopted":
        return f"{args[0]} adopts Christianity" if args else "Christianity adopted"
    if event_type == "rebel_outbreak":
        if len(args) >= 2:
            return f"Rebellion in {args[1]} — {args[0]}"
        return "Rebellion outbreak"
    if event_type == "rebel_crushed":
        if len(args) >= 2:
            return f"Rebellion crushed in {args[1]} — {args[0]}"
        return "Rebellion crushed"

    label = _smart_title(event_type)
    return label + (f" — {', '.join(args)}" if args else "")


def parse_event(event_node: Node, index: int) -> dict[str, Any]:
    if event_node.kind != "record" or not event_node.groups:
        raise ESFError("Malformed EVENT_LOG_DATA entry")

    children = event_node.groups[0]
    date_node = children[0] if children and children[0].kind == "record" else None
    date = parse_date_node(date_node)

    event_type = "unknown_event"
    for child in children:
        if child.kind == "utf16":
            event_type = str(child.value)
            break

    first_line = next(
        (
            x
            for x in children
            if x.kind == "record" and x.name == "FIRST_LINE_LOC_BLOCK"
        ),
        None,
    )
    second_line = next(
        (
            x
            for x in children
            if x.kind == "record" and x.name == "SECOND_LINE_LOC_BLOCK"
        ),
        None,
    )

    raw_args = loc_strings(first_line) + loc_strings(second_line)
    human_args = [humanize_loc(x) for x in raw_args]

    coords = [x.value for x in children[1:3] if x.kind == "i32"]

    return {
        "index": index,
        "year": date.get("year"),
        "season_index": date.get("season_index"),
        "season": date.get("season"),
        "date_raw": date.get("raw", []),
        "type": event_type,
        "category": event_category(event_type),
        "description": describe_event(event_type, raw_args),
        "args": human_args,
        "raw_localisation_keys": raw_args,
        "map_coordinates_raw": coords,
    }


def extract_events(player_faction: Node) -> list[dict[str, Any]]:
    event_log = child_record(player_faction, "EVENT_LOG")
    if not event_log or not event_log.groups or not event_log.groups[0]:
        return []

    container = event_log.groups[0][0]
    if container.kind != "record" or container.name != "EVENT_LOG_DATA":
        return []

    events: list[dict[str, Any]] = []
    for group in container.groups:
        if not group:
            continue
        candidate = group[0]
        if candidate.kind == "record" and candidate.name == "EVENT_LOG_DATA":
            events.append(parse_event(candidate, len(events)))
    return events


def extract_economics(player_faction: Node) -> dict[str, Any]:
    econ = child_record(player_faction, "FACTION_ECONOMICS")
    if not econ or not econ.groups:
        return {"treasury": None, "history": []}

    children = econ.groups[0]
    history_node = next(
        (x for x in children if x.kind == "record" and x.name == "history"), None
    )

    # In this Shogun 2 structure, the first scalar integer after the history
    # record is the current treasury.
    treasury = None
    history_seen = False
    for child in children:
        if child is history_node:
            history_seen = True
            continue
        if history_seen and child.kind in {"i32", "u32"}:
            treasury = int(child.value)
            break

    history: list[dict[str, Any]] = []
    if history_node:
        for idx, group in enumerate(history_node.groups):
            if not group:
                continue
            row = group[0]
            if row.kind != "record" or row.name != "ECONOMICS_DATA" or not row.groups:
                continue
            arrays = []
            for child in row.groups[0]:
                if child.kind.endswith("[]"):
                    val = list(child.value) if isinstance(child.value, (bytes, bytearray)) else child.value
                    arrays.append(val)
                else:
                    arrays.append(child.value)
            history.append(
                {
                    "stored_index": idx,
                    "record_version": row.version,
                    "raw_fields": arrays,
                }
            )

    trailing_scalars = [
        child.value
        for child in children
        if child.kind in {"i32", "u32", "i64", "u64"}
    ]

    return {
        "treasury": treasury,
        "history": history,
        "raw_scalar_fields": trailing_scalars,
        "note": (
            "The save stores rolling ECONOMICS_DATA rows, but this tool leaves "
            "their six array fields unlabeled until their exact Shogun 2 semantics "
            "are independently verified."
        ),
    }


def direct_array_group_count(player_faction: Node, record_name: str) -> Optional[int]:
    record = child_record(player_faction, record_name)
    if record is None:
        return None
    return len(record.groups)


def battle_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    battles = [x for x in events if x["type"] == "battle_fought"]
    wins = 0
    losses = 0
    unknown = 0
    opponents: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"wins": 0, "losses": 0, "unknown": 0, "battles": 0}
    )
    outcomes: collections.Counter[str] = collections.Counter()
    by_year: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"wins": 0, "losses": 0, "unknown": 0, "battles": 0}
    )

    for battle in battles:
        raw = battle["raw_localisation_keys"]
        opponent = battle["args"][0] if battle["args"] else "Unknown"
        result_key = raw[1] if len(raw) > 1 else ""
        result = battle["args"][1] if len(battle["args"]) > 1 else "Unknown"
        year = str(battle["year"])

        opponents[opponent]["battles"] += 1
        by_year[year]["battles"] += 1
        outcomes[result] += 1

        if "_victory_" in result_key:
            wins += 1
            opponents[opponent]["wins"] += 1
            by_year[year]["wins"] += 1
        elif "_defeat_" in result_key:
            losses += 1
            opponents[opponent]["losses"] += 1
            by_year[year]["losses"] += 1
        else:
            unknown += 1
            opponents[opponent]["unknown"] += 1
            by_year[year]["unknown"] += 1

    total_decided = wins + losses
    return {
        "battles": len(battles),
        "wins": wins,
        "losses": losses,
        "unknown": unknown,
        "win_rate": (wins / total_decided * 100.0) if total_decided else None,
        "by_opponent": dict(
            sorted(opponents.items(), key=lambda kv: (-kv[1]["battles"], kv[0]))
        ),
        "by_outcome": dict(outcomes.most_common()),
        "by_year": dict(sorted(by_year.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)),
    }


def territory_summary(events: list[dict[str, Any]], player_name: str) -> dict[str, Any]:
    captures = [x for x in events if x["type"] == "clan_region_taken"]
    player_captures = [
        x
        for x in captures
        if len(x["args"]) >= 2 and x["args"][1].casefold() == player_name.casefold()
    ]

    unique_player_regions = []
    seen = set()
    for event in player_captures:
        if event["args"]:
            region = event["args"][0]
            if region not in seen:
                seen.add(region)
                unique_player_regions.append(region)

    # Latest logged owner for each province that changed hands at least once.
    latest_owner: dict[str, str] = {}
    for event in captures:
        if len(event["args"]) >= 2:
            latest_owner[event["args"][0]] = event["args"][1]

    currently_last_logged_player = sorted(
        region
        for region, owner in latest_owner.items()
        if owner.casefold() == player_name.casefold()
    )

    return {
        "capture_events_total_world_log": len(captures),
        "player_capture_events": len(player_captures),
        "unique_regions_captured_by_player": unique_player_regions,
        "unique_player_capture_count": len(unique_player_regions),
        "regions_whose_latest_capture_event_names_player_as_owner": currently_last_logged_player,
        "latest_logged_player_region_count": len(currently_last_logged_player),
        "note": (
            "This is reconstructed from capture events. A starting province that never "
            "changed hands may not appear in this list, so it is not automatically the "
            "same as the save's total current province count."
        ),
    }


def extract_report(save_path: Path) -> dict[str, Any]:
    esf, compression = load_save(save_path)

    header = next(find_records(esf.root, "SAVE_GAME_HEADER"), None)
    if header is None or not header.groups:
        raise ESFError("SAVE_GAME_HEADER not found")

    hc = header.groups[0]
    faction_key = next((x.value for x in hc if x.kind == "utf16"), None)
    turn = hc[2].value if len(hc) > 2 and hc[2].kind in {"u32", "i32"} else None
    header_year = hc[3].value if len(hc) > 3 and hc[3].kind in {"u32", "i32"} else None
    header_season = hc[4].value if len(hc) > 4 and hc[4].kind == "utf16" else None
    header_date_node = next(
        (x for x in hc if x.kind == "record" and x.name == "DATE"), None
    )
    header_date = parse_date_node(header_date_node)

    factions = list(find_records(esf.root, "FACTION"))
    player_faction = None
    if faction_key:
        for faction in factions:
            if not faction.groups:
                continue
            if any(
                c.kind == "utf16" and c.value == faction_key
                for c in faction.groups[0]
            ):
                player_faction = faction
                break

    # Safe fallback: the faction with the largest non-empty EVENT_LOG.
    if player_faction is None:
        scored: list[tuple[int, Node]] = []
        for faction in factions:
            log = child_record(faction, "EVENT_LOG")
            size = 0
            if log and log.groups and log.groups[0]:
                container = log.groups[0][0]
                if container.kind == "record":
                    size = len(container.groups)
            scored.append((size, faction))
        if scored:
            player_faction = max(scored, key=lambda x: x[0])[1]

    if player_faction is None:
        raise ESFError("Could not identify the player faction")

    faction_name = humanize_loc(f"factions_screen_name_{faction_key}") if faction_key else "Unknown"
    events = extract_events(player_faction)
    economics = extract_economics(player_faction)

    counts = collections.Counter(event["type"] for event in events)
    categories = collections.Counter(event["category"] for event in events)

    first_event = events[0] if events else None
    last_event = events[-1] if events else None

    report = {
        "tool": {
            "name": "Shogun 2 Save Report",
            "version": "0.1.0",
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "read_only": True,
        },
        "source": {
            "filename": save_path.name,
            "path": str(save_path.resolve()),
            "size_bytes": save_path.stat().st_size,
            "format": esf.signature,
            "creation_date_raw": esf.creation_date,
            "compression": compression,
        },
        "campaign": {
            "faction_key": faction_key,
            "faction": faction_name,
            "turn": turn,
            "year": header_date.get("year") or header_year,
            "season_index": header_date.get("season_index"),
            "season": header_season or header_date.get("season"),
            "display_date": f"{header_season or header_date.get('season')} {header_date.get('year') or header_year}",
            "event_log_first_date": (
                f"{first_event['season']} {first_event['year']}" if first_event else None
            ),
            "event_log_last_date": (
                f"{last_event['season']} {last_event['year']}" if last_event else None
            ),
        },
        "current_snapshot": {
            "treasury": economics.get("treasury"),
            "army_record_count": direct_array_group_count(player_faction, "ARMY_ARRAY"),
            "character_record_count": direct_array_group_count(player_faction, "CHARACTER_ARRAY"),
            "note": (
                "Army/character counts are raw record counts in the faction arrays; "
                "they are not guaranteed to equal the number of currently deployable armies/agents."
            ),
        },
        "event_log": {
            "count": len(events),
            "type_counts": dict(counts.most_common()),
            "category_counts": dict(categories.most_common()),
            "events": events,
        },
        "battles": battle_summary(events),
        "territory": territory_summary(events, faction_name),
        "economics": economics,
    }
    return report


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_html(report: dict[str, Any]) -> str:
    campaign = report["campaign"]
    snapshot = report["current_snapshot"]
    battles = report["battles"]
    territory = report["territory"]
    event_log = report["event_log"]
    economics = report["economics"]
    events = event_log["events"]

    win_rate = battles["win_rate"]
    win_rate_text = f"{win_rate:.1f}%" if win_rate is not None else "—"

    opponent_rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{v['battles']}</td><td>{v['wins']}</td>"
        f"<td>{v['losses']}</td><td>{v['unknown']}</td></tr>"
        for name, v in battles["by_opponent"].items()
    ) or '<tr><td colspan="5">No recorded battles.</td></tr>'

    outcome_rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{count}</td></tr>"
        for name, count in battles["by_outcome"].items()
    ) or '<tr><td colspan="2">No recorded battle outcomes.</td></tr>'

    year_rows = "".join(
        f"<tr><td>{esc(year)}</td><td>{v['battles']}</td><td>{v['wins']}</td>"
        f"<td>{v['losses']}</td></tr>"
        for year, v in battles["by_year"].items()
    ) or '<tr><td colspan="4">No yearly battle data.</td></tr>'

    capture_rows = "".join(
        f"<li>{esc(region)}</li>" for region in territory["unique_regions_captured_by_player"]
    ) or "<li>No player capture events found.</li>"

    event_rows_parts = []
    for e in events:
        raw_keys = "<br>".join(esc(x) for x in e["raw_localisation_keys"])
        event_rows_parts.append(
            f'<tr class="event-row" data-category="{esc(e["category"])}" '
            f'data-search="{esc((e["description"] + " " + e["type"] + " " + " ".join(e["args"])).lower())}">'
            f'<td class="nowrap">{esc(e["season"])} {esc(e["year"])}</td>'
            f'<td><span class="pill">{esc(e["category"])}</span></td>'
            f'<td>{esc(e["description"])}</td>'
            f'<td><code>{esc(e["type"])}</code></td>'
            f'<td><details><summary>raw</summary><code>{raw_keys or "—"}</code></details></td>'
            "</tr>"
        )
    event_rows = "".join(event_rows_parts)

    econ_rows = []
    for row in economics["history"]:
        fields = row["raw_fields"]
        field_cells = "".join(f"<td><code>{esc(json.dumps(x))}</code></td>" for x in fields)
        econ_rows.append(
            f"<tr><td>{row['stored_index']}</td>{field_cells}</tr>"
        )
    max_fields = max((len(x["raw_fields"]) for x in economics["history"]), default=0)
    econ_headers = "".join(f"<th>Raw field {i + 1}</th>" for i in range(max_fields))
    econ_body = "".join(econ_rows) or '<tr><td colspan="7">No economic history found.</td></tr>'

    type_count_rows = "".join(
        f"<tr><td><code>{esc(name)}</code></td><td>{count}</td></tr>"
        for name, count in event_log["type_counts"].items()
    )

    categories = sorted(event_log["category_counts"].keys())
    category_options = "".join(
        f'<option value="{esc(c)}">{esc(c)}</option>' for c in categories
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(campaign['faction'])} — Shogun 2 Campaign Report</title>
<style>
:root {{ color-scheme: light dark; --bg:#111315; --panel:#191c20; --muted:#aeb6bf; --line:#333942; --accent:#d6a84b; --text:#f2f4f7; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); line-height:1.45; }}
main {{ max-width:1200px; margin:0 auto; padding:32px 20px 80px; }}
h1,h2,h3 {{ line-height:1.15; }} h1 {{ margin-bottom:6px; }} h2 {{ margin-top:36px; border-bottom:1px solid var(--line); padding-bottom:8px; }}
.muted {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:20px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; }}
.card .big {{ font-size:1.7rem; font-weight:750; color:var(--accent); }}
.card .label {{ color:var(--muted); font-size:.88rem; }}
table {{ width:100%; border-collapse:collapse; margin:12px 0 24px; }}
th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ position:sticky; top:0; background:var(--panel); z-index:1; }}
code {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.86em; overflow-wrap:anywhere; }}
.pill {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 8px; font-size:.78rem; color:var(--muted); }}
.controls {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0; }}
input,select {{ background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:7px; padding:8px 10px; }}
input {{ min-width:280px; flex:1; }}
.scroll {{ overflow:auto; max-height:720px; border:1px solid var(--line); border-radius:8px; }}
.scroll table {{ margin:0; }}
.nowrap {{ white-space:nowrap; }}
details summary {{ cursor:pointer; color:var(--muted); }}
.note {{ border-left:3px solid var(--accent); padding:8px 12px; background:var(--panel); color:var(--muted); }}
.two {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:22px; }}
ul.columns {{ columns:3 180px; }}
footer {{ margin-top:40px; color:var(--muted); font-size:.85rem; }}
@media (prefers-color-scheme: light) {{ :root {{ --bg:#f6f5f1; --panel:#fff; --muted:#59616a; --line:#d8d8d3; --accent:#8b5e00; --text:#181a1c; }} }}
</style>
</head>
<body><main>
<h1>{esc(campaign['faction'])} Campaign Report</h1>
<div class="muted">Total War: SHOGUN 2 · {esc(report['source']['filename'])} · internal date {esc(campaign['display_date'])}</div>

<div class="grid">
  <div class="card"><div class="big">{fmt_int(campaign['turn'])}</div><div class="label">Turn</div></div>
  <div class="card"><div class="big">{esc(campaign['display_date'])}</div><div class="label">Internal campaign date</div></div>
  <div class="card"><div class="big">{fmt_int(snapshot['treasury'])}</div><div class="label">Current treasury (koku)</div></div>
  <div class="card"><div class="big">{event_log['count']}</div><div class="label">Persistent event-log entries</div></div>
  <div class="card"><div class="big">{battles['wins']}–{battles['losses']}</div><div class="label">Recorded battle W–L</div></div>
  <div class="card"><div class="big">{win_rate_text}</div><div class="label">Recorded battle win rate</div></div>
</div>

<p class="note">The event log spans {esc(campaign['event_log_first_date'])} → {esc(campaign['event_log_last_date'])}. This is persistent campaign history, not merely the last few turns. The rolling economics block is separate and contains {len(economics['history'])} stored ECONOMICS_DATA rows.</p>

<h2>Battle record</h2>
<div class="two">
<div><h3>By opponent</h3><table><thead><tr><th>Opponent</th><th>Battles</th><th>Wins</th><th>Losses</th><th>?</th></tr></thead><tbody>{opponent_rows}</tbody></table></div>
<div><h3>By result</h3><table><thead><tr><th>Result</th><th>Count</th></tr></thead><tbody>{outcome_rows}</tbody></table></div>
</div>
<h3>By year</h3><table><thead><tr><th>Year</th><th>Battles</th><th>Wins</th><th>Losses</th></tr></thead><tbody>{year_rows}</tbody></table>

<h2>Territorial history</h2>
<div class="grid">
 <div class="card"><div class="big">{territory['player_capture_events']}</div><div class="label">Player capture events</div></div>
 <div class="card"><div class="big">{territory['unique_player_capture_count']}</div><div class="label">Unique provinces captured</div></div>
 <div class="card"><div class="big">{territory['latest_logged_player_region_count']}</div><div class="label">Changed provinces last logged as player-held</div></div>
</div>
<ul class="columns">{capture_rows}</ul>
<p class="muted">{esc(territory['note'])}</p>

<h2>Campaign event timeline</h2>
<div class="controls">
<input id="eventSearch" type="search" placeholder="Search events, clans, provinces, technologies…">
<select id="categoryFilter"><option value="">All categories</option>{category_options}</select>
</div>
<div class="scroll"><table id="events"><thead><tr><th>Date</th><th>Category</th><th>Event</th><th>Internal type</th><th>Source keys</th></tr></thead><tbody>{event_rows}</tbody></table></div>

<h2>Event-type counts</h2>
<table><thead><tr><th>Internal event type</th><th>Count</th></tr></thead><tbody>{type_count_rows}</tbody></table>

<h2>Rolling economic history</h2>
<p class="note">{esc(economics['note'])} Current treasury is decoded separately and is <strong>{fmt_int(economics['treasury'])} koku</strong>.</p>
<div class="scroll"><table><thead><tr><th>Stored row</th>{econ_headers}</tr></thead><tbody>{econ_body}</tbody></table></div>

<h2>Current-save structural snapshot</h2>
<table><tbody>
<tr><th>Faction</th><td>{esc(campaign['faction'])} (<code>{esc(campaign['faction_key'])}</code>)</td></tr>
<tr><th>Internal date</th><td>{esc(campaign['display_date'])}</td></tr>
<tr><th>Turn</th><td>{fmt_int(campaign['turn'])}</td></tr>
<tr><th>Treasury</th><td>{fmt_int(snapshot['treasury'])}</td></tr>
<tr><th>Army-array records</th><td>{fmt_int(snapshot['army_record_count'])}</td></tr>
<tr><th>Character-array records</th><td>{fmt_int(snapshot['character_record_count'])}</td></tr>
<tr><th>Save format</th><td>{esc(report['source']['format'])}</td></tr>
<tr><th>Compressed campaign section</th><td>{'Yes' if report['source']['compression']['compressed'] else 'No'}</td></tr>
</tbody></table>
<p class="muted">{esc(snapshot['note'])}</p>

<footer>Generated read-only by shogun2_report.py v{esc(report['tool']['version'])}. Raw/localisation identifiers are retained in the JSON output so later versions can improve decoding without losing source information.</footer>
</main>
<script>
const search = document.getElementById('eventSearch');
const category = document.getElementById('categoryFilter');
const rows = [...document.querySelectorAll('.event-row')];
function filterEvents() {{
  const q = search.value.trim().toLowerCase();
  const c = category.value;
  for (const row of rows) {{
    const okText = !q || row.dataset.search.includes(q);
    const okCat = !c || row.dataset.category === c;
    row.hidden = !(okText && okCat);
  }}
}}
search.addEventListener('input', filterEvents);
category.addEventListener('change', filterEvents);
</script>
</body></html>"""


def default_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".save"):
        name = name[:-5]
    return f"{name}_report"


def write_report(save_path: Path, out_dir: Path, write_json: bool = True) -> tuple[Path, Optional[Path], dict[str, Any]]:
    report = extract_report(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = default_stem(save_path)
    html_path = out_dir / f"{stem}.html"
    json_path = out_dir / f"{stem}.json" if write_json else None

    html_path.write_text(render_html(report), encoding="utf-8")
    if json_path:
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return html_path, json_path, report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a readable HTML/JSON report from Total War: SHOGUN 2 CAAB campaign saves."
    )
    parser.add_argument("saves", nargs="+", type=Path, help="One or more .save files")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: beside each input save)",
    )
    parser.add_argument("--no-json", action="store_true", help="Do not write the JSON companion file")
    args = parser.parse_args(argv)

    failures = 0
    for save in args.saves:
        try:
            if not save.is_file():
                raise FileNotFoundError(save)
            out_dir = args.out_dir if args.out_dir is not None else save.parent
            html_path, json_path, report = write_report(save, out_dir, not args.no_json)
            c = report["campaign"]
            b = report["battles"]
            print(f"OK: {save}")
            print(f"  {c['faction']} · {c['display_date']} · turn {c['turn']}")
            print(f"  events: {report['event_log']['count']} · battles: {b['wins']}-{b['losses']} · treasury: {report['current_snapshot']['treasury']}")
            print(f"  HTML: {html_path}")
            if json_path:
                print(f"  JSON: {json_path}")
        except Exception as exc:
            failures += 1
            print(f"ERROR: {save}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
