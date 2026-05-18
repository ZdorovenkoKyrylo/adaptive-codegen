from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict
from enum import Enum, auto
from abc import ABC, abstractmethod


class DataType(Enum):
    FLOAT32 = 'float'
    FLOAT64 = 'double'
    INT32 = 'int'
    INT64 = 'long'
    VOID = 'void'


@dataclass
class ASTNode(ABC):
    pass


@dataclass
class TypedVariable(ASTNode):
    name: str
    dtype: DataType
    shape: tuple = (1,)
    is_const: bool = False
    is_static: bool = False

    @property
    def c_type(self) -> str:
        base = self.dtype.value
        if self.is_const:
            base = f"const {base}"
        if self.is_static:
            base = f"static {base}"
        return base

    @property
    def c_declaration(self) -> str:
        if len(self.shape) == 1 and self.shape[0] == 1:
            return f"{self.c_type} {self.name}"
        elif len(self.shape) == 1:
            return f"{self.c_type} {self.name}[{self.shape[0]}]"
        else:
            dims = ''.join(f'[{d}]' for d in self.shape)
            return f"{self.c_type} {self.name}{dims}"


@dataclass
class Expression(ASTNode):
    pass


@dataclass
class Literal(Expression):
    value: Union[int, float, str]
    dtype: DataType = DataType.FLOAT64


@dataclass
class VariableRef(Expression):
    name: str
    indices: Optional[List] = None

    def to_c(self) -> str:
        if self.indices is None:
            return self.name
        idx_str = ''.join(
            f'[{idx}]' if isinstance(idx, str) else f'[{idx.to_c() if hasattr(idx, "to_c") else idx}]'
            for idx in self.indices
        )
        return f"{self.name}{idx_str}"


@dataclass
class BinaryOp(Expression):
    left: Expression
    op: str
    right: Expression


@dataclass
class UnaryOp(Expression):
    op: str
    operand: Expression


@dataclass
class FunctionCall(Expression):
    name: str
    arguments: List[Expression]


@dataclass
class Statement(ASTNode):
    pass


@dataclass
class Assignment(Statement):
    target: VariableRef
    value: Expression


@dataclass
class ForLoop(Statement):
    iterator: str
    start: Expression
    end: Expression
    step: Expression
    body: List[Statement]


@dataclass
class IfStatement(Statement):
    condition: Expression
    then_body: List[Statement]
    else_body: Optional[List[Statement]] = None


@dataclass
class WhileLoop(Statement):
    condition: Expression
    body: List[Statement]


@dataclass
class Return(Statement):
    value: Optional[Expression] = None


@dataclass
class FunctionDef(ASTNode):
    name: str
    return_type: DataType
    parameters: List[TypedVariable]
    body: List[Statement]
    is_inline: bool = False
    is_static: bool = False


@dataclass
class StructDef(ASTNode):
    name: str
    fields: List[TypedVariable]


@dataclass
class Module(ASTNode):
    name: str
    includes: List[str]
    structs: List[StructDef]
    globals: List[TypedVariable]
    functions: List[FunctionDef]


class ASTBuilder:
    @staticmethod
    def var(name: str, dtype: DataType = DataType.FLOAT64,
            shape: tuple = (1,)) -> TypedVariable:
        return TypedVariable(name=name, dtype=dtype, shape=shape)

    @staticmethod
    def ref(name: str, *indices) -> VariableRef:
        idx_list = list(indices) if indices else None
        return VariableRef(name=name, indices=idx_list)

    @staticmethod
    def lit(value: Union[int, float]) -> Literal:
        dtype = DataType.INT32 if isinstance(value, int) else DataType.FLOAT64
        return Literal(value=value, dtype=dtype)

    @staticmethod
    def add(left: Expression, right: Expression) -> BinaryOp:
        return BinaryOp(left=left, op='+', right=right)

    @staticmethod
    def mul(left: Expression, right: Expression) -> BinaryOp:
        return BinaryOp(left=left, op='*', right=right)

    @staticmethod
    def call(name: str, *args) -> FunctionCall:
        return FunctionCall(name=name, arguments=list(args))

    @staticmethod
    def assign(target: VariableRef, value: Expression) -> Assignment:
        return Assignment(target=target, value=value)

    @staticmethod
    def for_loop(iterator: str, n: int, body: List[Statement]) -> ForLoop:
        return ForLoop(
            iterator=iterator,
            start=Literal(0),
            end=Literal(n),
            step=Literal(1),
            body=body
        )
