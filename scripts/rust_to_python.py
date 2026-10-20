#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Colm Talbot. waveformjax follows LALSuite, GSL and
# waveformrs: see NOTICE.md for the original code and its authors.
"""Translate the numerical Rust of waveformrs into Python for waveformjax.

waveformrs (``../waveformrs``) writes LALSimulation's parameter-space fits,
coefficient tables and closed-form expressions as straight-line Rust: ``let``
bindings, arithmetic, ``fma`` calls and the methods of ``f64``. This script
parses that subset (items, statements and expressions) and emits the same
code as Python on JAX arrays, so that the thousands of fitted numbers are
copied by a program rather than by hand:

* ``fma(a, b, c)``, ``fms``, ``fnms`` stay function calls (``a * b + c`` ...
  in :mod:`waveformjax.models.xpnr._math`);
* ``x.sqrt()``, ``x.powf(y)``, ``x.libm_cbrt()``, ``y.atan2(x)``, ... become
  ``jnp`` functions;
* ``const`` arrays become nested tuples of floats;
* ``if`` expressions become ``jnp.where`` (both branches are evaluated,
  which is what a traced JAX function needs anyway);
* everything the script does not understand is emitted as a ``raise
  NotImplementedError`` line with the Rust source in a comment, to be
  translated by hand.

Usage::

    python scripts/rust_to_python.py ../waveformrs/src/imrphenomx/pnr_fits.rs \\
        --functions alpha_a1 beta_b0 --consts COEFF_A1 > out.py

The generated modules of :mod:`waveformjax.models.xpnr` say at the top which
command produced them (``scripts/generate_fits.py`` runs them all).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------

PUNCT = [
    "..=", "...", "::", "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=",
    "%=", "..", "<<", ">>",
    "{", "}", "(", ")", "[", "]", ";", ",", ".", ":", "=", "<", ">", "+", "-", "*", "/", "%",
    "!", "&", "|", "#", "?", "'", "@", "^",
]

NUMBER = re.compile(
    r"(0x[0-9a-fA-F_]+|(\d[\d_]*)(\.(?![.a-zA-Z_])\d*[\d_]*)?([eE][+-]?\d+)?)(f64|f32|usize|u32|i32|u64|i64|isize)?"
)
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class Token:
    kind: str  # "num", "id", "p", "str", "eof"
    text: str
    pos: int


def tokenize(src: str) -> List[Token]:
    tokens = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            depth = 0
            while i < n:
                if src.startswith("/*", i):
                    depth += 1
                    i += 2
                elif src.startswith("*/", i):
                    depth -= 1
                    i += 2
                    if depth == 0:
                        break
                else:
                    i += 1
            continue
        if c == '"' or (c == "b" and i + 1 < n and src[i + 1] == '"'):
            j = i + (2 if c == "b" else 1)
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            tokens.append(Token("str", src[i : j + 1], i))
            i = j + 1
            continue
        if c == "'" and i + 2 < n and src[i + 2] == "'":
            tokens.append(Token("str", src[i : i + 3], i))
            i += 3
            continue
        if c == "'" and IDENT.match(src, i + 1):
            # lifetime
            m = IDENT.match(src, i + 1)
            i = m.end()
            continue
        if c.isdigit():
            m = NUMBER.match(src, i)
            tokens.append(Token("num", m.group(0), i))
            i = m.end()
            continue
        m = IDENT.match(src, i)
        if m:
            tokens.append(Token("id", m.group(0), i))
            i = m.end()
            continue
        for p in PUNCT:
            if src.startswith(p, i):
                tokens.append(Token("p", p, i))
                i += len(p)
                break
        else:
            raise SyntaxError(f"unexpected character {c!r} at {i}")
    tokens.append(Token("eof", "", n))
    return tokens


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------


@dataclass
class Node:
    pass


@dataclass
class Num(Node):
    value: str


@dataclass
class Name(Node):
    path: str


@dataclass
class Unary(Node):
    op: str
    operand: Node


@dataclass
class Binary(Node):
    op: str
    left: Node
    right: Node


@dataclass
class Call(Node):
    func: Node
    args: List[Node]


@dataclass
class Method(Node):
    obj: Node
    name: str
    args: List[Node]


@dataclass
class Field(Node):
    obj: Node
    name: str


@dataclass
class Index(Node):
    obj: Node
    index: Node


@dataclass
class ArrayLit(Node):
    items: List[Node]


@dataclass
class ArrayRepeat(Node):
    item: Node
    count: Node


@dataclass
class TupleLit(Node):
    items: List[Node]


@dataclass
class Cast(Node):
    expr: Node
    to: str


@dataclass
class IfExpr(Node):
    cond: Node
    then: "Block"
    other: Optional[Node]  # Block or IfExpr


@dataclass
class Closure(Node):
    params: List[str]
    body: Node


@dataclass
class StructLit(Node):
    name: str
    fields: List[tuple]
    base: Optional[Node]


@dataclass
class RangeExpr(Node):
    start: Optional[Node]
    end: Optional[Node]
    inclusive: bool


@dataclass
class Raw(Node):
    text: str


@dataclass
class Block(Node):
    stmts: List[Node]
    tail: Optional[Node]


@dataclass
class Let(Node):
    pattern: Node
    value: Optional[Node]


@dataclass
class Assign(Node):
    op: str
    target: Node
    value: Node


@dataclass
class ExprStmt(Node):
    expr: Node


@dataclass
class For(Node):
    pattern: Node
    iterable: Node
    body: Block


@dataclass
class While(Node):
    cond: Node
    body: Block


@dataclass
class Return(Node):
    value: Optional[Node]


@dataclass
class Function(Node):
    name: str
    params: List[str]
    body: Block
    doc: List[str] = field(default_factory=list)
    owner: Optional[str] = None


@dataclass
class Const(Node):
    name: str
    value: Node
    doc: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

BINARY_PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "==": 3, "!=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3,
    "|": 4, "^": 5, "&": 6, "<<": 7, ">>": 7,
    "+": 8, "-": 8,
    "*": 9, "/": 9, "%": 9,
}


class Parser:
    def __init__(self, src: str):
        self.src = src
        self.tokens = tokenize(src)
        self.i = 0
        self.no_struct = 0

    # helpers
    def peek(self, k=0) -> Token:
        return self.tokens[min(self.i + k, len(self.tokens) - 1)]

    def next(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def at(self, text, k=0) -> bool:
        t = self.peek(k)
        return t.text == text and t.kind in ("p", "id")

    def expect(self, text) -> Token:
        t = self.next()
        if t.text != text:
            ctx = self.src[max(0, t.pos - 80) : t.pos + 40]
            raise SyntaxError(f"expected {text!r}, got {t.text!r} near: {ctx!r}")
        return t

    def skip_balanced(self, open_, close):
        depth = 0
        while True:
            t = self.next()
            if t.text == open_:
                depth += 1
            elif t.text == close:
                depth -= 1
                if depth == 0:
                    return
            if t.kind == "eof":
                raise SyntaxError("unbalanced")

    def skip_type(self):
        """Skip a type (after ':' or '->' or 'as')."""
        depth = 0
        while True:
            t = self.peek()
            if t.kind == "eof":
                return
            if t.text in ("<", "(", "["):
                depth += 1
            elif t.text in (">", ")", "]"):
                if depth == 0:
                    return
                depth -= 1
            elif t.text == ">>":
                depth -= 2
                if depth < 0:
                    return
            elif depth == 0 and t.text in (",", "=", "{", ";", "where", "|"):
                return
            self.next()

    # items
    def parse_items(self):
        items = []
        doc = []
        owner = []
        while self.peek().kind != "eof":
            t = self.peek()
            if t.text == "#":
                self.next()
                if self.at("!"):
                    self.next()
                self.skip_balanced("[", "]")
                continue
            if t.text == "}" and owner:
                self.next()
                owner.pop()
                continue
            if t.text in ("pub",):
                self.next()
                if self.at("("):
                    self.skip_balanced("(", ")")
                continue
            if t.text in ("const", "static") and self.peek(1).kind == "id" and self.peek(1).text != "fn":
                self.next()
                name = self.next().text
                self.expect(":")
                self.skip_type()
                self.expect("=")
                value = self.parse_expr()
                self.expect(";")
                items.append(Const(name, value, doc))
                doc = []
                continue
            if t.text == "const" and self.at("fn", 1):
                self.next()
                continue
            if t.text in ("unsafe", "extern", "async") and not self.at("{", 1):
                self.next()
                if self.peek().kind == "str":
                    self.next()
                continue
            if t.text == "fn":
                start = self.i
                try:
                    f = self.parse_function()
                except (SyntaxError, IndexError) as error:
                    # skip the function: translate it by hand
                    self.i = start
                    while not self.at("{"):
                        self.next()
                    self.skip_balanced("{", "}")
                    print(f"# could not parse fn {self.tokens[start + 1].text}: {error}"[:200], file=sys.stderr)
                    continue
                f.owner = owner[-1] if owner else None
                items.append(f)
                continue
            if t.text == "impl":
                self.next()
                if self.at("<"):
                    self.skip_angle()
                name = []
                while not self.at("{"):
                    name.append(self.next().text)
                self.next()
                owner.append(" ".join(name))
                continue
            if t.text in ("mod", "use", "type", "struct", "enum", "trait", "macro_rules", "extern"):
                # skip to the end of the item
                self.next()
                depth = 0
                while True:
                    u = self.next()
                    if u.text == "{":
                        depth += 1
                    elif u.text == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    elif u.text == ";" and depth == 0:
                        break
                    if u.kind == "eof":
                        break
                continue
            self.next()
        return items

    def skip_angle(self):
        depth = 0
        while True:
            t = self.next()
            if t.text == "<":
                depth += 1
            elif t.text == ">":
                depth -= 1
                if depth == 0:
                    return
            elif t.text == ">>":
                depth -= 2
                if depth <= 0:
                    return

    def parse_function(self) -> Function:
        start = self.peek().pos
        self.expect("fn")
        name = self.next().text
        if self.at("<"):
            self.skip_angle()
        self.expect("(")
        params = []
        while not self.at(")"):
            if self.at("&"):
                self.next()
                if self.at("mut"):
                    self.next()
            if self.at("mut"):
                self.next()
            t = self.next()
            if t.text == "self":
                params.append("self")
            elif t.text in ("(", "["):
                # destructuring parameter
                close = ")" if t.text == "(" else "]"
                names = []
                while not self.at(close):
                    u = self.next()
                    if u.kind == "id" and u.text != "mut":
                        names.append(u.text)
                self.next()
                params.append("(" + ", ".join(names) + ")")
                self.expect(":")
                self.skip_type()
            else:
                params.append(t.text)
                self.expect(":")
                self.skip_type()
            if self.at(","):
                self.next()
        self.expect(")")
        if self.at("->"):
            self.next()
            self.skip_type()
        if self.at("where"):
            while not self.at("{"):
                self.next()
        body = self.parse_block()
        f = Function(name, params, body)
        f.source = self.src[start : self.tokens[self.i - 1].pos + 1]
        return f

    # statements
    def parse_block(self) -> Block:
        self.expect("{")
        stmts = []
        tail = None
        saved = self.no_struct
        self.no_struct = 0
        while not self.at("}"):
            if self.at(";"):
                self.next()
                continue
            if self.at("#"):
                self.next()
                self.skip_balanced("[", "]")
                continue
            if self.at("const") or self.at("static"):
                self.next()
                name = self.next().text
                self.expect(":")
                self.skip_type()
                self.expect("=")
                value = self.parse_expr()
                self.expect(";")
                stmts.append(Let(Name(name), value))
                continue
            if self.at("type"):
                while not self.at(";"):
                    self.next()
                self.next()
                continue
            if self.at("let"):
                self.next()
                pattern = self.parse_pattern()
                if self.at(":"):
                    self.next()
                    self.skip_type()
                value = None
                if self.at("="):
                    self.next()
                    value = self.parse_expr()
                if self.at("else"):
                    self.next()
                    self.parse_block()
                    value = Raw("let-else")
                self.expect(";")
                stmts.append(Let(pattern, value))
                continue
            if self.at("for"):
                self.next()
                pattern = self.parse_pattern()
                self.expect("in")
                self.no_struct += 1
                iterable = self.parse_expr()
                self.no_struct -= 1
                body = self.parse_block()
                stmts.append(For(pattern, iterable, body))
                continue
            if self.at("while"):
                self.next()
                self.no_struct += 1
                cond = self.parse_expr()
                self.no_struct -= 1
                body = self.parse_block()
                stmts.append(While(cond, body))
                continue
            if self.at("return"):
                self.next()
                value = None if self.at(";") or self.at("}") else self.parse_expr()
                stmts.append(Return(value))
                continue
            if self.at("if") or self.at("{") or self.at("loop") or self.at("match"):
                if self.at("loop"):
                    start = self.peek().pos
                    self.next()
                    self.skip_balanced("{", "}")
                    node = Raw("LOOP<" + self.src[start : self.tokens[self.i - 1].pos + 1] + ">")
                else:
                    node = self.parse_if() if self.at("if") else (self.parse_block() if self.at("{") else self.parse_primary())
                if self.at("}"):
                    tail = node
                    break
                if self.at(";"):
                    self.next()
                if not (self.peek().kind == "p" and self.peek().text in (".", "?")):
                    stmts.append(ExprStmt(node))
                    continue
                expr = self.parse_postfix(node)
            else:
                expr = self.parse_expr()
            if self.peek().text in ("=", "+=", "-=", "*=", "/=", "%="):
                op = self.next().text
                value = self.parse_expr()
                stmts.append(Assign(op, expr, value))
                if self.at(";"):
                    self.next()
                continue
            if self.at(";"):
                self.next()
                stmts.append(ExprStmt(expr))
                continue
            if self.at("}"):
                tail = expr
                break
            # block-like expression statements (if/for without ';')
            stmts.append(ExprStmt(expr))
        self.expect("}")
        self.no_struct = saved
        return Block(stmts, tail)

    def parse_pattern(self) -> Node:
        if self.at("mut"):
            self.next()
        if self.at("&"):
            self.next()
        if self.at("("):
            self.next()
            items = []
            while not self.at(")"):
                items.append(self.parse_pattern())
                if self.at(","):
                    self.next()
            self.next()
            return TupleLit(items)
        if self.at("["):
            self.next()
            items = []
            while not self.at("]"):
                items.append(self.parse_pattern())
                if self.at(","):
                    self.next()
            self.next()
            return ArrayLit(items)
        t = self.next()
        name = t.text
        # struct patterns are not supported
        return Name(name)

    # expressions
    def parse_expr(self, min_prec=0) -> Node:
        left = self.parse_unary()
        while True:
            t = self.peek()
            if t.kind == "p" and t.text in ("..", "..=") and min_prec == 0:
                self.next()
                end = None
                if not (self.at(")") or self.at("]") or self.at("{") or self.at(",") or self.at(";")):
                    end = self.parse_expr(1)
                left = RangeExpr(left, end, t.text == "..=")
                continue
            if t.kind == "id" and t.text == "as":
                self.next()
                ty = self.next().text
                left = Cast(left, ty)
                continue
            prec = BINARY_PRECEDENCE.get(t.text) if t.kind == "p" else None
            if prec is None or prec <= min_prec:
                return left
            self.next()
            right = self.parse_expr(prec)
            left = Binary(t.text, left, right)

    def parse_unary(self) -> Node:
        t = self.peek()
        if t.kind == "p" and t.text in ("..", "..="):
            self.next()
            end = self.parse_expr(1)
            return RangeExpr(None, end, t.text == "..=")
        if t.text in ("-", "!") and t.kind == "p":
            self.next()
            operand = self.parse_unary()
            return Unary(t.text, operand)
        if t.text == "&" and t.kind == "p":
            self.next()
            if self.at("mut"):
                self.next()
            return self.parse_unary()
        if t.text == "*" and t.kind == "p":
            self.next()
            return self.parse_unary()
        return self.parse_postfix(self.parse_primary())

    def parse_args(self, close=")") -> List[Node]:
        args = []
        saved = self.no_struct
        self.no_struct = 0
        while not self.at(close):
            args.append(self.parse_expr())
            if self.at(","):
                self.next()
        self.next()
        self.no_struct = saved
        return args

    def parse_postfix(self, node: Node) -> Node:
        while True:
            if self.at("."):
                self.next()
                t = self.next()
                name = t.text
                if self.at("::"):
                    # turbofish
                    self.next()
                    self.skip_angle()
                if self.at("("):
                    self.next()
                    node = Method(node, name, self.parse_args())
                else:
                    node = Field(node, name)
                continue
            if self.at("("):
                self.next()
                node = Call(node, self.parse_args())
                continue
            if self.at("["):
                self.next()
                saved = self.no_struct
                self.no_struct = 0
                index = self.parse_expr()
                self.no_struct = saved
                self.expect("]")
                node = Index(node, index)
                continue
            if self.at("?"):
                self.next()
                continue
            return node

    def parse_primary(self) -> Node:
        t = self.peek()
        if t.kind == "num":
            self.next()
            return Num(t.text)
        if t.kind == "str":
            self.next()
            return Raw(t.text)
        if t.text == "(":
            self.next()
            saved = self.no_struct
            self.no_struct = 0
            if self.at(")"):
                self.next()
                self.no_struct = saved
                return TupleLit([])
            first = self.parse_expr()
            if self.at(","):
                items = [first]
                while self.at(","):
                    self.next()
                    if self.at(")"):
                        break
                    items.append(self.parse_expr())
                self.expect(")")
                self.no_struct = saved
                return TupleLit(items)
            self.expect(")")
            self.no_struct = saved
            return Unary("()", first)
        if t.text == "[":
            self.next()
            saved = self.no_struct
            self.no_struct = 0
            items = []
            if self.at("]"):
                self.next()
                self.no_struct = saved
                return ArrayLit([])
            first = self.parse_expr()
            if self.at(";"):
                self.next()
                count = self.parse_expr()
                self.expect("]")
                self.no_struct = saved
                return ArrayRepeat(first, count)
            items.append(first)
            while self.at(","):
                self.next()
                if self.at("]"):
                    break
                items.append(self.parse_expr())
            self.expect("]")
            self.no_struct = saved
            return ArrayLit(items)
        if t.text == "{":
            return self.parse_block()
        if t.text == "if":
            return self.parse_if()
        if t.text in ("|", "||"):
            return self.parse_closure()
        if t.text == "move":
            self.next()
            return self.parse_closure()
        if t.text == "match":
            start = t.pos
            self.next()
            self.no_struct += 1
            self.parse_expr()
            self.no_struct -= 1
            self.skip_balanced("{", "}")
            return Raw("MATCH<" + self.src[start : self.tokens[self.i - 1].pos + 1] + ">")
        if t.text == "unsafe":
            self.next()
            return self.parse_block()
        if t.kind == "id":
            self.next()
            path = t.text
            while self.at("::"):
                self.next()
                if self.at("<"):
                    self.skip_angle()
                    continue
                path += "::" + self.next().text
            if self.at("!"):
                # macro
                self.next()
                start = self.peek().pos
                open_ = self.peek().text
                close = {"(": ")", "[": "]", "{": "}"}[open_]
                self.skip_balanced(open_, close)
                return Raw(f"MACRO<{path}!{self.src[start:self.tokens[self.i - 1].pos + 1]}>")
            if self.at("{") and not self.no_struct and (path[0].isupper() or path == "Self"):
                return self.parse_struct(path)
            return Name(path)
        raise SyntaxError(f"unexpected token {t.text!r} near {self.src[max(0, t.pos - 60): t.pos + 30]!r}")

    def parse_struct(self, name) -> Node:
        self.expect("{")
        fields = []
        base = None
        saved = self.no_struct
        self.no_struct = 0
        while not self.at("}"):
            if self.at(".."):
                self.next()
                base = self.parse_expr()
                continue
            fname = self.next().text
            if self.at(":"):
                self.next()
                value = self.parse_expr()
            else:
                value = Name(fname)
            fields.append((fname, value))
            if self.at(","):
                self.next()
        self.next()
        self.no_struct = saved
        return StructLit(name, fields, base)

    def parse_if(self) -> Node:
        self.expect("if")
        self.no_struct += 1
        if self.at("let"):
            start = self.peek().pos
            while not self.at("{"):
                self.next()
            cond = Raw("IFLET<" + self.src[start : self.peek().pos] + ">")
        else:
            cond = self.parse_expr()
        self.no_struct -= 1
        then = self.parse_block()
        other = None
        if self.at("else"):
            self.next()
            if self.at("if"):
                other = self.parse_if()
            else:
                other = self.parse_block()
        return IfExpr(cond, then, other)

    def parse_closure(self) -> Node:
        params = []
        if self.at("||"):
            self.next()
        else:
            self.expect("|")
            while not self.at("|"):
                if self.at("&"):
                    self.next()
                if self.at("mut"):
                    self.next()
                p = self.parse_pattern()
                params.append(emit_pattern(p))
                if self.at(":"):
                    self.next()
                    self.skip_type()
                if self.at(","):
                    self.next()
            self.next()
        if self.at("->"):
            self.next()
            self.skip_type()
        body = self.parse_expr()
        return Closure(params, body)


# --------------------------------------------------------------------------
# emitter
# --------------------------------------------------------------------------

CONSTANTS = {
    "PI": "PI",
    "std::f64::consts::PI": "PI",
    "FRAC_PI_2": "FRAC_PI_2",
    "FRAC_PI_4": "FRAC_PI_4",
    "LN_2": "LN_2",
    "f64::MIN_POSITIVE": "DBL_MIN",
    "f64::EPSILON": "DBL_EPSILON",
    "f64::NAN": "NAN",
    "f64::INFINITY": "INF",
    "f64::NEG_INFINITY": "(-INF)",
    "true": "True",
    "false": "False",
}

UNARY_METHODS = {
    "sqrt": "jnp.sqrt",
    "ln": "jnp.log",
    "exp": "jnp.exp",
    "abs": "jnp.abs",
    "libm_cbrt": "jnp.cbrt",
    "cbrt": "jnp.cbrt",
    "sin": "jnp.sin",
    "cos": "jnp.cos",
    "tan": "jnp.tan",
    "atan": "jnp.arctan",
    "acos": "jnp.arccos",
    "asin": "jnp.arcsin",
    "tanh": "jnp.tanh",
    "sinh": "jnp.sinh",
    "cosh": "jnp.cosh",
    "log10": "jnp.log10",
    "log2": "jnp.log2",
    "floor": "jnp.floor",
    "ceil": "jnp.ceil",
    "round": "jnp.round",
    "is_nan": "jnp.isnan",
    "is_finite": "jnp.isfinite",
    "signum": "jnp.sign",
    "exp_m1": "jnp.expm1",
    "ln_1p": "jnp.log1p",
    "trunc": "jnp.trunc",
}

PYTHON_PREC = {
    "or": 1, "and": 2, "not": 3,
    "==": 4, "!=": 4, "<": 4, ">": 4, "<=": 4, ">=": 4,
    "|": 5, "^": 6, "&": 7, "<<": 8, ">>": 8,
    "+": 9, "-": 9, "*": 10, "/": 10, "%": 10, "//": 10,
    "unary": 11, "**": 12, "atom": 13,
}

OP_MAP = {"&&": "&", "||": "|"}


class Untranslatable(Exception):
    pass


def emit_pattern(p: Node) -> str:
    if isinstance(p, Name):
        return "_" if p.path == "_" else p.path
    if isinstance(p, (TupleLit, ArrayLit)):
        return "(" + ", ".join(emit_pattern(x) for x in p.items) + ("," if len(p.items) == 1 else "") + ")"
    raise Untranslatable("pattern")


def number(text: str) -> str:
    text = text.replace("_", "")
    text = re.sub(r"(f64|f32|usize|u32|i32|u64|i64|isize)$", "", text)
    if text.endswith("."):
        text += "0"
    return text


class Emitter:
    def __init__(self, cast_ints=True):
        self.cast_ints = cast_ints

    def expr(self, node: Node) -> tuple:
        """(text, precedence)"""
        if isinstance(node, Num):
            return number(node.value), PYTHON_PREC["atom"]
        if isinstance(node, Name):
            path = node.path
            if path in CONSTANTS:
                return CONSTANTS[path], PYTHON_PREC["atom"]
            if path.startswith("Self::"):
                path = path[len("Self::"):]
            return path.replace("::", "."), PYTHON_PREC["atom"]
        if isinstance(node, Raw):
            raise Untranslatable(node.text)
        if isinstance(node, Unary):
            if node.op == "()":
                text, _ = self.expr(node.operand)
                return f"({text})", PYTHON_PREC["atom"]
            text, prec = self.expr(node.operand)
            if node.op == "-":
                if prec < PYTHON_PREC["unary"]:
                    text = f"({text})"
                return f"-{text}", PYTHON_PREC["unary"]
            return f"jnp.logical_not({text})", PYTHON_PREC["atom"]
        if isinstance(node, Binary):
            op = OP_MAP.get(node.op, node.op)
            p = PYTHON_PREC[op]
            lt, lp = self.expr(node.left)
            rt, rp = self.expr(node.right)
            if op in ("&", "|"):
                # logical operators on arrays
                return f"({lt}) {op} ({rt})", PYTHON_PREC["atom"] - 1
            if lp < p:
                lt = f"({lt})"
            if rp < p or (rp == p and op in ("-", "/", "%", "==", "!=", "<", ">", "<=", ">=")):
                rt = f"({rt})"
            return f"{lt} {op} {rt}", p
        if isinstance(node, Call):
            func, _ = self.expr(node.func)
            args = ", ".join(self.expr(a)[0] for a in node.args)
            if func in ("Some", "Ok"):
                return args, PYTHON_PREC["atom"]
            if func == "Complex.new":
                re_, im = (self.expr(a)[0] for a in node.args)
                return f"({re_} + 1j * ({im}))", PYTHON_PREC["atom"]
            return f"{func}({args})", PYTHON_PREC["atom"]
        if isinstance(node, Method):
            obj, op = self.expr(node.obj)
            if op < PYTHON_PREC["atom"]:
                obj = f"({obj})"
            name = node.name
            args = [self.expr(a)[0] for a in node.args]
            if name in UNARY_METHODS and not args:
                return f"{UNARY_METHODS[name]}({obj})", PYTHON_PREC["atom"]
            if name == "powf":
                return f"jnp.power({obj}, {args[0]})", PYTHON_PREC["atom"]
            if name == "powi":
                return f"{obj} ** {args[0]}", PYTHON_PREC["**"] - 1
            if name == "atan2":
                return f"jnp.arctan2({obj}, {args[0]})", PYTHON_PREC["atom"]
            if name == "copysign":
                return f"jnp.copysign({obj}, {args[0]})", PYTHON_PREC["atom"]
            if name == "max":
                return f"jnp.maximum({obj}, {args[0]})", PYTHON_PREC["atom"]
            if name == "min":
                return f"jnp.minimum({obj}, {args[0]})", PYTHON_PREC["atom"]
            if name == "hypot":
                return f"jnp.hypot({obj}, {args[0]})", PYTHON_PREC["atom"]
            if name == "mul_add":
                return f"fma({obj}, {args[0]}, {args[1]})", PYTHON_PREC["atom"]
            if name in ("clone", "to_owned", "into", "unwrap", "as_ref"):
                return obj, op
            if name == "sin_cos":
                return f"(jnp.sin({obj}), jnp.cos({obj}))", PYTHON_PREC["atom"]
            return f"{obj}.{name}({', '.join(args)})", PYTHON_PREC["atom"]
        if isinstance(node, Field):
            obj, op = self.expr(node.obj)
            if op < PYTHON_PREC["atom"]:
                obj = f"({obj})"
            if node.name.isdigit():
                return f"{obj}[{node.name}]", PYTHON_PREC["atom"]
            return f"{obj}.{node.name}", PYTHON_PREC["atom"]
        if isinstance(node, Index):
            obj, op = self.expr(node.obj)
            if op < PYTHON_PREC["atom"]:
                obj = f"({obj})"
            index = node.index
            if isinstance(index, RangeExpr):
                start = self.expr(index.start)[0] if index.start else ""
                end = self.expr(index.end)[0] if index.end else ""
                if index.inclusive:
                    end = f"{end} + 1"
                return f"{obj}[{start}:{end}]", PYTHON_PREC["atom"]
            return f"{obj}[{self.expr(index)[0]}]", PYTHON_PREC["atom"]
        if isinstance(node, ArrayLit):
            return "[" + ", ".join(self.expr(x)[0] for x in node.items) + "]", PYTHON_PREC["atom"]
        if isinstance(node, ArrayRepeat):
            return f"[{self.expr(node.item)[0]}] * {self.expr(node.count)[0]}", PYTHON_PREC["*"]
        if isinstance(node, TupleLit):
            items = [self.expr(x)[0] for x in node.items]
            return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")", PYTHON_PREC["atom"]
        if isinstance(node, Cast):
            text, prec = self.expr(node.expr)
            if node.to in ("f64", "f32"):
                return text, prec
            if self.cast_ints:
                return f"int({text})", PYTHON_PREC["atom"]
            return text, prec
        if isinstance(node, IfExpr):
            return self.if_expr(node), PYTHON_PREC["atom"]
        if isinstance(node, Block):
            if node.stmts or node.tail is None:
                raise Untranslatable("block expression")
            return self.expr(node.tail)
        if isinstance(node, Closure):
            return f"(lambda {', '.join(node.params)}: {self.expr(node.body)[0]})", PYTHON_PREC["atom"]
        if isinstance(node, StructLit):
            fields = ", ".join(f"{name}={self.expr(value)[0]}" for name, value in node.fields)
            if node.base is not None:
                raise Untranslatable("struct update")
            return f"Struct({fields})", PYTHON_PREC["atom"]
        if isinstance(node, RangeExpr):
            start = self.expr(node.start)[0] if node.start else "0"
            end = self.expr(node.end)[0]
            if node.inclusive:
                end = f"{end} + 1"
            return f"range({start}, {end})", PYTHON_PREC["atom"]
        raise Untranslatable(type(node).__name__)

    def if_expr(self, node: IfExpr) -> str:
        cond = self.expr(node.cond)[0]
        then = self.expr(node.then)[0]
        if node.other is None:
            raise Untranslatable("if without else")
        other = self.expr(node.other)[0]
        return f"jnp.where({cond}, {then}, {other})"

    # statements
    def block(self, block: Block, indent: str, is_function_body=False) -> List[str]:
        lines = []
        for stmt in block.stmts:
            lines.extend(self.stmt(stmt, indent))
        if block.tail is not None:
            if is_function_body:
                lines.extend(self.tail_return(block.tail, indent))
            else:
                lines.extend(self.stmt(ExprStmt(block.tail), indent))
        if not lines:
            lines.append(indent + "pass")
        return lines

    def tail_return(self, tail: Node, indent: str) -> List[str]:
        if isinstance(tail, IfExpr):
            try:
                return [indent + "return " + self.expr(tail)[0]]
            except Untranslatable:
                return self.if_stmt(tail, indent, returns=True)
        return [indent + "return " + self.expr(tail)[0]]

    def if_stmt(self, node: IfExpr, indent: str, returns=False) -> List[str]:
        lines = [indent + "# TODO(jax): Python control flow on a traced value?"]
        lines.append(indent + f"if {self.expr(node.cond)[0]}:")
        body = self.block(node.then, indent + "    ", is_function_body=returns)
        lines.extend(body)
        other = node.other
        while isinstance(other, IfExpr):
            lines.append(indent + f"elif {self.expr(other.cond)[0]}:")
            lines.extend(self.block(other.then, indent + "    ", is_function_body=returns))
            other = other.other
        if other is not None:
            lines.append(indent + "else:")
            lines.extend(self.block(other, indent + "    ", is_function_body=returns))
        return lines

    def stmt(self, stmt: Node, indent: str) -> List[str]:
        try:
            if isinstance(stmt, Let):
                if stmt.value is None:
                    return [indent + f"{emit_pattern(stmt.pattern)} = None"]
                return [indent + f"{emit_pattern(stmt.pattern)} = {self.expr(stmt.value)[0]}"]
            if isinstance(stmt, Assign):
                target = self.expr(stmt.target)[0]
                if isinstance(stmt.target, TupleLit):
                    target = target[1:-1]
                return [indent + f"{target} {stmt.op} {self.expr(stmt.value)[0]}"]
            if isinstance(stmt, Return):
                if stmt.value is None:
                    return [indent + "return"]
                return [indent + "return " + self.expr(stmt.value)[0]]
            if isinstance(stmt, For):
                iterable = self.expr(stmt.iterable)[0]
                lines = [indent + f"for {emit_pattern(stmt.pattern)} in {iterable}:"]
                lines.extend(self.block(stmt.body, indent + "    "))
                return lines
            if isinstance(stmt, While):
                lines = [indent + "# TODO(jax): while loop", indent + f"while {self.expr(stmt.cond)[0]}:"]
                lines.extend(self.block(stmt.body, indent + "    "))
                return lines
            if isinstance(stmt, ExprStmt):
                if isinstance(stmt.expr, IfExpr):
                    return self.if_stmt(stmt.expr, indent)
                if isinstance(stmt.expr, Block):
                    return self.block(stmt.expr, indent)
                return [indent + self.expr(stmt.expr)[0]]
        except Untranslatable as error:
            return [indent + f"raise NotImplementedError('untranslated Rust: {str(error)[:60]!s}')".replace("\n", " ")]
        raise Untranslatable(type(stmt).__name__)

    def function(self, f: Function, name: Optional[str] = None, docstring: Optional[str] = None) -> str:
        params = [p for p in f.params]
        lines = [f"def {name or f.name}({', '.join(params)}):"]
        if docstring:
            lines.append(f'    """{docstring}"""')
        try:
            lines.extend(self.block(f.body, "    ", is_function_body=True))
        except Untranslatable as error:
            lines.append(f"    raise NotImplementedError('untranslated: {error}')")
        return "\n".join(lines)

    def const_value(self, node: Node) -> str:
        """A table as nested tuples of Python floats."""
        if isinstance(node, ArrayLit):
            return "(" + ", ".join(self.const_value(x) for x in node.items) + ("," if len(node.items) == 1 else "") + ")"
        if isinstance(node, ArrayRepeat):
            return f"({self.const_value(node.item)},) * {self.expr(node.count)[0]}"
        if isinstance(node, Unary) and node.op == "-" and isinstance(node.operand, Num):
            return "-" + number(node.operand.value)
        return self.expr(node)[0]


def doc_comment(src: str, pos: int) -> str:
    """The /// comment block right above position `pos`."""
    lines = src[:pos].rstrip().split("\n")
    doc = []
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("#["):
            continue
        if s.startswith("pub") or s in ("", "const", "unsafe"):
            continue
        if s.startswith("///"):
            doc.append(s[3:].strip())
        else:
            break
    return " ".join(reversed(doc))


def parse_file(path):
    src = open(path).read()
    parser = Parser(src)
    return src, parser.parse_items()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("--functions", nargs="*", default=None)
    parser.add_argument("--consts", nargs="*", default=None)
    parser.add_argument("--owner", default=None, help="only methods of this impl block")
    args = parser.parse_args(argv)
    src, items = parse_file(args.source)
    emitter = Emitter()
    out = []
    for item in items:
        if isinstance(item, Function):
            if args.functions is not None and item.name not in args.functions:
                continue
            if args.owner is not None and item.owner != args.owner:
                continue
            out.append(emitter.function(item))
        elif isinstance(item, Const):
            if args.consts is not None and item.name not in args.consts:
                continue
            out.append(f"{item.name} = {emitter.const_value(item.value)}")
    print("\n\n\n".join(out))


if __name__ == "__main__":
    main()
