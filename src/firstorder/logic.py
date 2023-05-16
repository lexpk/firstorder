from dataclasses import dataclass
from typing import Tuple, Union
from lark import Lark, ParseTree
from itertools import combinations
from sys import maxsize


@dataclass(eq=True, frozen=True)
class Function:
    name: str
    args: Tuple['Term']

    def __repr__(self) -> str:
        if len(self.args) == 0:
            return self.name
        else:
            return "{}({})".format(
                self.name,
                ", ".join(repr(a) for a in self.args),
            )

    def __lt__(self, other: "Term"):
        if type(other) == Function:
            return (self.name, self.args) < (other.name, other.args)
        if type(other) == Variable:
            return False

    def _normalize_variables(self, mapping: dict['Variable', 'Variable']):
        args = []
        for a in self.args:
            args.append(a._normalize_variables(mapping))
        return Function(self.name, tuple(args))


@dataclass(eq=True, frozen=True)
class Variable:
    id: int

    def __repr__(self) -> str:
        return "X{}".format(self.id)

    def __lt__(self, other: "Term"):
        if type(other) == Function:
            return True
        if type(other) == Variable:
            return self.id < other.id

    def _normalize_variables(self, mapping: dict['Variable', 'Variable']):
        if self not in mapping:
            mapping[self] = Variable(len(mapping))
        return mapping[self]


Term = Function | Variable


def is_subterm(subterm: Term, term: Term) -> bool:
    """Checks if a term is a subterm of another term.

    Parameters
    ----------
    Term : Term
        The term.
    subterm : Term
        The subterm.

    Returns
    -------
    bool
        True if the subterm is a subterm of the term, False otherwise.

    Example
    -------
    >>> t = Function("f", (Variable(0), Variable(1)))
    >>> is_subterm(t, t)
    True
    >>> is_subterm(Variable(0), t)
    True
    >>> is_subterm(Variable(1), t)
    True
    >>> is_subterm(Function("f", (Variable(0), Variable(1))), t)
    True
    >>> is_subterm(Function("f", (Variable(0), Variable(0))), t)
    False
    """
    if subterm == term:
        return True
    if isinstance(term, Function):
        for a in term.args:
            if is_subterm(subterm, a):
                return True
    return False


def get_subterm(Term, index: Tuple[int, ...]):
    """Gets a subterm of a term.

    Parameters
    ----------
    Term : Term
        The term.
    index : Tuple[int]
        The index of the subterm.

    Returns
    -------
    Term
        The subterm.

    Example
    -------
    >>> t = Function("f", (Variable(0), Variable(1)))
    >>> subterm(t, (0,))
    Variable(0)
    >>> subterm(t, (1,))
    Variable(1)
    >>> subterm(t, (0, 0))
    Traceback (most recent call last):
    ...
    TypeError: 'Variable' object is not subscriptable
    """
    for i in index:
        Term = Term.args[i]
    return Term


def replace_subterm(s: Term, index: tuple[int, ...], t: Term):
    """Replaces a subterm of a term.

    Parameters
    ----------
    s : Term
        The term.
    index : tuple[int, ...]
        The index of the subterm.
    t : Term
        The term to replace the subterm with.

    Returns
    -------
    Term
        The term with the subterm replaced.

    Example
    -------
    >>> t = Function("f", (Variable(0), Variable(1)))
    >>> replace_subterm(t, (0,), Variable(2))
    Function('f', (Variable(2), Variable(1)))
    >>> replace_subterm(t, (1,), Variable(2))
    Function('f', (Variable(0), Variable(2)))
    """
    if len(index) == 0:
        return t
    elif len(index) == 1:
        if isinstance(s, Function):
            return Function(s.name, tuple(
                t if i == index[0] else a
                for i, a in enumerate(s.args)
            ))
        else:
            raise TypeError("'{}' object is not subscriptable".format(
                type(s).__name__
            ))
    else:
        if isinstance(s, Function):
            return Function(s.name, tuple(
                replace_subterm(a, index[1:], t)
                if i == index[0] else a
                for i, a in enumerate(s.args)
            ))
        else:
            raise TypeError("'{}' object is not subscriptable".format(
                type(s).__name__
            ))


def subtermindices(term: Term) -> list[tuple[int, ...]]:
    """Returns a list of all subterm indices for the given term.

    Parameters
    ----------
    term : Term
        The term to get the subterm indices for.

    Returns
    -------
    list[Tuple]
        The list of subterm indices.
    """
    if isinstance(term, Variable):
        return [tuple()]
    elif isinstance(term, Function):
        indices = [tuple()]
        for i, arg in enumerate(term.args):
            for subindex in subtermindices(arg):
                indices.append((i,) + subindex)
        return indices
    else:
        raise ValueError(
            "Invalid term type: {}. Must be\
            Variable or Function.".format(type(term))
        )


class Equation:
    """Represents an equation."""

    def __init__(self, left: Term, right: Term):
        """Initializes an equation.

        Parameters
        ----------
        left : Term
            The left side of the equation.
        right : Term
            The right side of the equation.
        """
        if left < right:
            self.left = left
            self.right = right
        else:
            self.left = right
            self.right = left

    def __eq__(self, other):
        return (self.left, self.right) == (other.left, other.right)

    def __repr__(self) -> str:
        return "{} = {}".format(self.left, self.right)

    def __lt__(self, other):
        return (self.left, self.right) < (other.left, other.right)

    def __hash__(self):
        return hash((self.left, self.right))

    def is_trivial(self) -> bool:
        """Checks if the equation is trivial.

        Returns
        -------
        bool
            True if the equation is trivial, False otherwise.

        Example
        -------
        >>> Equation(Variable(0), Variable(0)).is_trivial()
        True
        >>> Equation(Variable(0), Variable(1)).is_trivial()
        False
        """
        return self.left == self.right

    def replace_subterm(self, side: str, index: tuple[int, ...], t: Term):
        """Replaces a subterm of a side of the equation.

        Parameters
        ----------
        side : str
            The side of the equation to replace the subterm on.
        index : tuple[int, ...]
            The index of the subterm.
        t : Term
            The term to replace the subterm with.

        Returns
        -------
        Equation
            The equation with the subterm replaced.

        Example
        -------
        >>> e = Equation(Function("f", (Variable(0), Variable(1))), Variable(2))
        >>> e.replace_subterm("left", (0,), Variable(2))
        Equation(Function('f', (Variable(2), Variable(1))), Variable(2))
        >>> e.replace_subterm("left", (1,), Variable(2))
        Equation(Function('f', (Variable(0), Variable(2))), Variable(2))
        >>> e.replace_subterm("right", tuple(), Variable(3))
        Equation(Function('f', (Variable(0), Variable(1))), Variable(3))
        """
        if side == "l":
            return Equation(
                replace_subterm(self.left, index, t),
                self.right
            )
        elif side == "r":
            return Equation(
                self.left,
                replace_subterm(self.right, index, t)
            )
        else:
            raise ValueError("Invalid side: {}".format(side))

    def _normalize_variables(self, mapping: dict[Variable, Variable]) -> 'Equation':
        """Normalizes the variables in the equation.

        Parameters
        ----------
        mapping : dict[Variable, Variable]
            The mapping of variables to normalized variables.

        Returns
        -------
        Equation
            The normalized equation.
        """
        left = self.left._normalize_variables(mapping)
        right = self.right._normalize_variables(mapping)
        return Equation(left, right)


class Sequent:
    """Represents a sequent.

    Attributes
    ----------
    left : Tuple[Equation]
        The antecedent of the sequent.
    right : Tuple[Equation]
        The succedent of the sequent.
    """

    def __init__(self, left: tuple[Equation], right: tuple[Equation]):
        """Initializes a sequent.

        Parameters
        ----------
        left : Tuple[Equation]
            The antecedent of the sequent.
        right : Tuple[Equation]
            The succedent of the sequent.
        """
        self.left = tuple(sorted(set(left)))
        self.right = tuple(sorted(set(right)))

    def __eq__(self, other):
        return (
            self.left, self.right
        ) == (
            other.left, other.right
        )

    def __repr__(self) -> str:
        return "{} -> {}".format(
            ", ".join(repr(e) for e in self.left),
            ", ".join(repr(e) for e in self.right),
        )

    def __lt__(self, other):
        return (
            self.left, self.right
        ) < (
            other.left, other.right
        )

    def __hash__(self):
        return hash((self.left, self.right))

    def is_trivial(self) -> bool:
        """Checks if the succedent contains a trivial equation.

        Returns
        -------
        bool
            True if the succedent contains a trivial equation, False otherwise.
        """
        return any(e.is_trivial() for e in self.right)

    def is_empty(self) -> bool:
        """Checks if the sequent is empty.

        Returns
        -------
        bool
            True if the sequent is empty, False otherwise.
        """
        return len(self.left) == 0 and len(self.right) == 0

    def replace_subterm(self, side: str, index: int, equation_side: str, equation_index: tuple[int, ...], t: Term):
        """Replaces a subterm of a side of an equation in the sequent.

        Parameters
        ----------
        side : str
            The side of the sequent to replace the subterm on.
        index : int
            The index of the equation.
        equation_side : str
            The side of the equation to replace the subterm on.
        equation_index : tuple[int, ...]
            The index of the subterm.
        t : Term
            The term to replace the subterm with.

        Returns
        -------
        Sequent
            The sequent with the subterm replaced.

        Example
        -------
        >>> s = Sequent(
        ...     (Equation(Function("f", (Variable(0), Variable(1))), Variable(2)),),
        ...     (Equation(Variable(3), Variable(4)),)
        ... )
        >>> s.replace_subterm("l", 0, "l", (0,), Variable(2))
        Sequent((Equation(Function('f', (Variable(2), Variable(1))),
                Variable(2)),), (Equation(Variable(3), Variable(4)),))
        >>> s.replace_subterm("l", 0, "l", (1,), Variable(2))
        Sequent((Equation(Function('f', (Variable(0), Variable(2))),
                Variable(2)),), (Equation(Variable(3), Variable(4)),))
        >>> s.replace_subterm("l", 0, "r", tuple(), Variable(3))
        Sequent((Equation(Function('f', (Variable(0), Variable(1))),
                Variable(3)),), (Equation(Variable(3), Variable(4)),))
        >>> s.replace_subterm("r", 0, "l", tuple(), Variable(3))
        Sequent((Equation(Function('f', (Variable(0), Variable(1))),
                Variable(2)),), (Equation(Variable(3), Variable(4)),))
        >>> s.replace_subterm("r", 0, "r", tuple(), Variable(3))
        Sequent((Equation(Function('f', (Variable(0), Variable(1))),
                Variable(2)),), (Equation(Variable(3), Variable(3)),))
        """
        if side == "l":
            return Sequent(
                tuple(
                    e if i != index else e.replace_subterm(
                        equation_side, equation_index, t)
                    for i, e in enumerate(self.left)
                ),
                self.right
            )
        elif side == "r":
            return Sequent(
                self.left,
                tuple(
                    e if i != index else e.replace_subterm(
                        equation_side, equation_index, t)
                    for i, e in enumerate(self.right)
                )
            )
        else:
            raise ValueError("Invalid side: {}".format(side))

    def remove_equation(self, side: str, index: int):
        """Produces a new sequent with an equation removed.

        Parameters
        ----------
        side : str
            The side of the sequent to remove the equation from.
        index : int
            The index of the equation.

        Returns
        -------
        Sequent
            The sequent with the equation removed.

        Example
        -------
        >>> s = Sequent(
        ...     (Equation(Function("f", (Variable(0), Variable(1))), Variable(2)),),
        ...     (Equation(Variable(3), Variable(4)),)
        ... )
        >>> s.remove_equation("l", 0)
        Sequent((), (Equation(Variable(3), Variable(4)),))
        >>> s.remove_equation("r", 0)
        Sequent((Equation(Function('f', (Variable(0), Variable(1))),
                Variable(2)),), ())
        """
        if side == "l":
            return Sequent(
                tuple(
                    e for i, e in enumerate(self.left) if i != index
                ),
                self.right
            )
        elif side == "r":
            return Sequent(
                self.left,
                tuple(
                    e for i, e in enumerate(self.right) if i != index
                )
            )
        else:
            raise ValueError("Invalid side: {}".format(side))

    def replace_equation(self, side: str, index: int, equation: Equation):
        """Produces a new sequent with an equation replaced.

        Parameters
        ----------
        side : str
            The side of the sequent to replace the equation on.
        index : int
            The index of the equation.
        equation : Equation
            The equation to replace the equation with.

        Returns
        -------
        Sequent
            The sequent with the equation replaced.

        Example
        -------
        >>> s = Sequent(
        ...     (Equation(Function("f", (Variable(0), Variable(1))), Variable(2)),),
        ...     (Equation(Variable(3), Variable(4)),)
        ... )
        >>> s.replace_equation("l", 0, Equation(Variable(0), Variable(1)))
        Sequent((Equation(Variable(0), Variable(1)),),
                (Equation(Variable(3), Variable(4)),))
        >>> s.replace_equation("r", 0, Equation(Variable(0), Variable(1)))
        Sequent((Equation(Function('f', (Variable(0), Variable(1))),
                Variable(2)),), (Equation(Variable(0), Variable(1)),))
        """
        if side == "l":
            return Sequent(
                tuple(
                    equation if i == index else e
                    for i, e in enumerate(self.left)
                ),
                self.right
            )
        elif side == "r":
            return Sequent(
                self.left,
                tuple(
                    equation if i == index else e
                    for i, e in enumerate(self.right)
                )
            )
        else:
            raise ValueError("Invalid side: {}".format(side))

    def equality_resolution(self) -> 'Sequent':
        """Applies equality resolution to the sequent.

        Returns
        -------
        Sequent
            The sequent resulting from applying equality resolution.
        """
        for i, equation in enumerate(self.left):
            unifier = mgu(equation.left, equation.right, disjoint=False)
            if unifier is not None:
                return substitute(
                    unifier,
                    self.remove_equation("l", i)
                )
        return self

    def equality_factoring(self) -> 'Sequent':
        """Applies equality factoring to the sequent.

        Returns
        -------
        Sequent
            The sequent resulting from applying equality factoring.
        """
        for (e1, e2) in filter(
            lambda m: m[0] != m[1],
            combinations(self.right, 2)
        ):
            for (t1, t2, t3, t4) in (
                (e1.left, e1.right, e2.left, e2.right),
                (e1.right, e1.left, e2.left, e2.right),
                (e1.left, e1.right, e2.right, e2.left),
                (e1.right, e1.left, e2.right, e2.left)
            ):
                unifier = mgu(t1, t3, disjoint=False)
                if unifier is not None:
                    return Sequent(
                        tuple(substitute(
                            unifier, e
                        ) for e in self.left) +
                        (Equation(
                            substitute(unifier, t2),
                            substitute(unifier, t4),
                        ),),
                        tuple(substitute(
                            unifier, e
                        ) for e in self.right if e != e1 and e != e2) +
                        (Equation(
                            substitute(unifier, t1),
                            substitute(unifier, t4),
                        ),),
                    )
        return self

    def normalize(self) -> 'Sequent':
        """Normalizes the sequent.

        Returns
        -------
        Sequent
            The normalized sequent.
        """
        new_sequent = self
        new_sequent.right = tuple(equation for equation in new_sequent.right if equation != Equation(
            Function("false", tuple()), Function("true", tuple())))
        new_sequent.right = tuple(sorted([equation for equation in set(
            new_sequent.right) if equation.left != equation.right]))
        new_sequent.left = tuple(sorted([equation for equation in set(
            new_sequent.left) if equation.left != equation.right]))
        return new_sequent.normalize_variables()

    def normalize_variables(self) -> 'Sequent':
        """Normalizes the variables in the sequent.

        Parameters
        ----------
        mapping : dict[Variable, Variable]
            The mapping from variables to variables.

        Returns
        -------
        Sequent
            The sequent with the variables normalized.
        """
        mapping = {}
        return Sequent(
            tuple(e._normalize_variables(mapping) for e in self.left),
            tuple(e._normalize_variables(mapping) for e in self.right)
        )


Substitution = dict[Variable, Term]


def compose_substitutions(s1: Substitution, s2: Substitution) -> Substitution:
    """Composes two substitutions.

    Parameters
    ----------
    s1: Substitution
        The first substitution.
    s2: Substitution
        The second substitution.

    Returns
    -------
    Substitution
        The composition of the two substitutions.

    Example
    -------
    >> > s1 = {Variable(0): Variable(1)}
    >> > s2 = {Variable(1): Variable(2)}
    >> > compose_substitutions(s1, s2)
    {Variable(0): Variable(2)}
    """
    return {k: substitute(s2, v) for k, v in s1.items()}


def mgu(
    t1: Term,
    t2: Term,
    disjoint: bool = True
) -> Union[Substitution, tuple[Substitution, Substitution]]:
    """Computes the most general unifier of two terms.

    Parameters
    ----------
    t1: Term
        The first term.
    t2: Term
        The second term.
    disjoint: bool, optional
        If True treats variables as disjoint, even if they have the same id.
        By default True.

    Returns
    -------
    Union[Substitution, tuple[Substitution, Substitution]]
        The most general unifier of the two terms.
        None if the terms cannot be unified.
        If disjoint is True, returns a pair of substitutions.

    Example
    -------
    >> > t1 = Function("f", (Variable(0),))
    >> > t2 = Function("f", (Variable(1),))
    >> > mgu(t1, t2)
    ({Variable(0): Variable(1)}, {})
    """
    if isinstance(t1, Variable):
        if isinstance(t2, Variable):
            if t1 == t2:
                if disjoint:
                    return (dict(), dict())
                else:
                    return dict()
            else:
                if disjoint:
                    return ({t1: t2}, dict())
                else:
                    return dict()
        else:
            if disjoint:
                return ({t1: t2}, dict())
            else:
                if is_subterm(t1, t2):
                    return None
                else:
                    return {t1: t2}
    elif isinstance(t2, Variable):
        if disjoint:
            return (dict(), {t2: t1})
        else:
            if is_subterm(t2, t1):
                return None
            else:
                return {t2: t1}
    else:
        if t1.name != t2.name:
            return None
        elif len(t1.args) != len(t2.args):
            return None
        else:
            if disjoint:
                s1 = dict()
                s2 = dict()
                for a1, a2 in zip(t1.args, t2.args):
                    m = mgu(
                        substitute(s1, a1),
                        substitute(s2, a2),
                        disjoint
                    )
                    if m is None:
                        return None
                    s1.update(m[0])
                    s2.update(m[1])
                    for k, v in s1.items():
                        s1[k] = substitute(s2, v)
                    for k, v in s2.items():
                        s2[k] = substitute(s1, v)
                return (s1, s2)
            else:
                s = dict()
                for a1, a2 in zip(t1.args, t2.args):
                    m = mgu(
                        substitute(s, a1),
                        substitute(s, a2),
                        disjoint
                    )
                    if m is None:
                        return None
                    s.update(m)
                return s


def substitute(
    s: Substitution,
    x: Union[Term, Equation, Sequent]
) -> Union[Term, Equation, Sequent]:
    """Substitutes a term for a variable.

    Parameters
    ----------
    s: Substitution
        The substitution.
    x: Union[Term, Equation, Sequent]
        The term, equation or sequent.

    Returns
    -------
    Union[Term, Equation, Sequent]
        The term, equation or sequent with the substitution applied.

    Example
    -------
    >> > s = {Variable(0): Variable(1)}
    >> > x = Equation(Variable(0), Variable(2))
    >> > substitute(s, x)
    Equation(Variable(1), Variable(2))
    """
    if isinstance(x, Equation):
        return Equation(substitute(s, x.left), substitute(s, x.right))
    elif isinstance(x, Sequent):
        return Sequent(
            tuple(substitute(s, e) for e in x.left),
            tuple(substitute(s, e) for e in x.right)
        )
    else:
        if isinstance(x, Variable):
            if x in s:
                return s[x]
            else:
                return x
        else:
            return Function(
                x.name,
                tuple(substitute(s, a) for a in x.args)
            )


class Problem():
    """An environment for first-order logic.

    Attributes
    ----------
    functions: list[str]
        The functions that have been declared.
    arities: dict[str, int]
        A mapping from functions to their arities.
    axioms: list[Sequent]
        The axioms of the problem.
    conjectures: list[Sequent]
        The conjectures of the problem.
    variablecounter: int
        The number of variables that have been declared.

    Methods
    -------
    declare_function(name: str, arity: int)
        Declares a new function.
    read_sequent(s: str)
        Parses a sequent from a string.
    """

    def __init__(self):
        self.functions: list[str] = list()
        self.arity: dict[str, int] = dict()
        self.axioms: list[Sequent] = list()
        self.conjectures: list[Sequent] = list()
        self.variablecounter: int = 0

    def __repr__(self):
        return "\n".join(
            f"{i}.\t{s}" for i, s in enumerate(self.axioms)
        )

    def declare_function(
        self,
        name: str,
        arity: int
    ):
        """Declares a new function.

        Parameters
        ----------
        name: str
            The name of the function.
        arity: int
            The arity of the function.

        Raises
        ------
        Exception
            If the function already exists with a different arity.

        Example
        -------
        >> > p = Problem()
        >> > p.declare_function("f", 2)
        >> > p.declare_function("g", 1)
        """
        if name in self.functions:
            if self.arity[name] != arity:
                raise Exception(f"Function {name} already exists!")
        else:
            self.functions.append(name)
            self.arity[name] = arity
            newline = "\n"
            singleback = "\\"
            doubleback = "\\\\"
            self.grammar = Lark(f"""
                %import common.WS
                %ignore WS
                %import common.CNAME
                sequent: equations "->" equations
                equations: (equation ("," equation)*)?
                equation: term "=" term
                term: {" | ".join(f'''f{i}''' for i, f in enumerate(
                    self.functions))} | variable
                {newline.join(f'''f{i}: "{f.replace(singleback, doubleback)}" arguments''' for i, f in enumerate(
                    self.functions))}
                arguments: ("(" term ("," term)* ")")?
                variable: ("?")? CNAME
            """, start="sequent")

    def read_axiom(self, s: str):
        """Parses a axiom from a string.

        Parameters
        ----------
        s: str
            The string to parse.

        Example:
        >>> p = Problem()
        >>> p.declare_function("f", 2)
        >>> p.read_axiom("-> f(x, y) = f(y, x)")
        >>> p.read_axiom("f(x) = x ->")
        >>> p.read_axiom("f(x, y) = f(y, x) -> f(x) = x")
        """
        try:
            tree = self.grammar.parse(s)
        except Exception:
            raise Exception(f"Cannot parse sequent {s}")
        sequent = self._parse_sequent(tree, [])
        if sequent not in self.axioms:
            self.axioms.append(sequent.normalize())

    def read_conjecture(self, s: str):
        """Parses a negated conjecture from a string.

        Parameters
        - ---------
        s: str
            The string to parse.

        Example:
        >>> p = Problem()
        >>> p.declare_function("f", 2)
        >>> p.read_conjecture("-> f(x, y) = f(y, x)")
        >>> p.read_conjecture("f(x) = x ->")
        >>> p.read_conjecture("f(x, y) = f(y, x) -> f(x) = x")
        """
        try:
            tree = self.grammar.parse(s)
        except Exception:
            raise Exception(f"Cannot parse sequent {s}")
        sequent = self._parse_sequent(tree, [])
        if sequent not in self.conjectures:
            self.conjectures.append(sequent.normalize())

    def _parse_sequent(self, tree: ParseTree, variables) -> Sequent:
        left = self._parse_equations(tree.children[0], variables)
        right = self._parse_equations(tree.children[1], variables)
        return Sequent(left, right)

    def _parse_equations(self, tree, variables) -> Tuple[Equation]:
        return tuple(self._parse_equation(e, variables) for e in tree.children)

    def _parse_equation(self, tree, variables) -> Equation:
        return Equation(
            self._parse_term(tree.children[0], variables),
            self._parse_term(tree.children[1], variables)
        )

    def _parse_term(self, tree, variables) -> Term:
        if tree.children[0].data.value == "variable":
            if tree.children[0].children[0].value not in variables:
                variables.append(tree.children[0].children[0].value)
            return Variable(variables.index(tree.children[0].children[0].value))
        else:
            return Function(
                self.functions[int(tree.children[0].data.value[1:])],
                self._parse_args(tree.children[0].children[0], variables)
            )

    def _parse_args(self, tree, variables) -> Tuple[Term]:
        if tree is None:
            return tuple()
        else:
            return tuple(self._parse_term(t, variables) for t in tree.children)


class TermInstance():
    """An instance of a term in a sequent.

    Parameters
    - ---------
    sequent: Sequent
        The sequent in which the term occurs.
    sequent_side: str
        The side of the sequent in which the term occurs.
    equation_index: int
        The index of the equation in which the term occurs.
    equation_side: str
        The side of the equation in which the term occurs.
    subterm_index: tuple[int, ...]
        The index of the subterm in which the term occurs.

    """

    def __init__(
        self,
        sequent: Sequent,
        sequent_side: str,
        equation_index: int,
        equation_side: str,
        subterm_index: tuple[int, ...]
    ):
        self.sequent = sequent
        self.sequent_side = sequent_side
        self.equation_index = equation_index
        self.equation_side = equation_side
        self.subterm_index = subterm_index

    def __repr__(self):
        return "TermInstance({}, {}, {}, {}, {})".format(
            self.sequent,
            self.sequent_side,
            self.equation_index,
            self.equation_side,
            self.subterm_index,
        )

    def __eq__(self, other):
        return (
            self.sequent_index == other.sequent_index
            and self.sequent_side == other.sequent_side
            and self.equation_index == other.equation_index
            and self.equation_side == other.equation_side
            and self.subterm_index == other.subterm_index
        )

    def __lt__(self, other):
        return (
            self.sequent_index,
            self.sequent_side,
            self.equation_index,
            self.equation_side,
            self.subterm_index
        ) < (
            other.sequent_index,
            other.sequent_side,
            other.equation_index,
            other.equation_side,
            other.subterm_index
        )

    def equation(self) -> Equation:
        """Returns the equation corresponding to the term instance.

        Returns
        - ------
        Equation
            The equation corresponding to the sequent instance.
        """
        if self.sequent_side == "l":
            eqs = self.sequent.left
        elif self.sequent_side == "r":
            eqs = self.sequent.right
        else:
            raise ValueError(
                "Invalid sequent side: {}. Must be\
                'l' or 'r'.".format(self.sequent_side)
            )
        return eqs[self.equation_index]

    def toplevel(self) -> Term:
        """Returns the top-level term corresponding to the term instance.

        Returns
        - ------
        Term
            The top-level term corresponding to the term instance.
        """
        eq = self.equation()
        if self.equation_side == "l":
            return eq.left
        elif self.equation_side == "r":
            return eq.right
        else:
            raise ValueError(
                "Invalid equation side: {}. Must be\
                'l' or 'r'.".format(self.equation_side)
            )

    def term(self) -> Term:
        """Returns the subterm corresponding to the term instance.

        Returns
        - ------
        Term
            The subterm corresponding to the term instance.
        """
        term = self.toplevel()
        return get_subterm(term, self.subterm_index)


def positive_toplevel_terminstances(sequent: Sequent) -> list[TermInstance]:
    """Given a sequent, returns a list of all positive top level term

    Parameters
    - ---------
    sequent: Sequent
        The sequent to get the positive top level term instances from .

    Returns
    - ------
    list[TermInstance]
        The list of term instances.
    """
    terminstances = []
    for j, eq in enumerate(sequent.right):
        for side in ["l", "r"]:
            terminstances.append(
                TermInstance(sequent, "r", j, side, tuple())
            )
    return terminstances


def terminstances(sequent: Sequent) -> list[TermInstance]:
    """Given a sequent, returns a list of all term instances in the sequent.

    Parameters
    - ---------
    sequent: Sequent
        The sequent to get the term instances from .

    Returns
    - ------
    list[TermInstance]
        A list of all term instances in the sequent.
    """
    terminstances = []
    for j, eq in enumerate(sequent.left):
        for subindex in subtermindices(eq.left):
            terminstances.append(
                TermInstance(sequent, "l", j, "l", subindex)
            )
        for subindex in subtermindices(eq.right):
            terminstances.append(
                TermInstance(sequent, "l", j, "r", subindex)
            )
    for j, eq in enumerate(sequent.right):
        for subindex in subtermindices(eq.left):
            terminstances.append(
                TermInstance(sequent, "r", j, "l", subindex)
            )
        for subindex in subtermindices(eq.right):
            terminstances.append(
                TermInstance(sequent, "r", j, "r", subindex)
            )
    return terminstances


def superposition(
    toplevel: TermInstance,
    terminstance: TermInstance
) -> Sequent:
    """Given two term instances, returns the sequent obtained by superposition.

    Parameters
    - ---------
    toplevel: TermInstance
        The top level term instance.
    terminstance: TermInstance
        The term instance to superpose.

    Returns
    - ------
    Sequent
        The sequent obtained by superposition.
    """
    m = mgu(toplevel.term(), terminstance.term(), disjoint=True)
    if m is None:
        return None
    else:
        m0, m1 = m
    new_toplevel = substitute(m0, toplevel.sequent.remove_equation(
        toplevel.sequent_side, toplevel.equation_index
    ))
    new_equation = terminstance.equation().replace_subterm(
        terminstance.equation_side,
        terminstance.subterm_index,
        substitute(
            m0,
            toplevel.equation().left if toplevel.equation_side == "r" else toplevel.equation().right
        )
    )
    new_terminstance = substitute(
        m1,
        terminstance.sequent.replace_equation(
            terminstance.sequent_side, terminstance.equation_index, new_equation
        ),
    )
    return Sequent(
        new_toplevel.left + new_terminstance.left,
        new_toplevel.right + new_terminstance.right
    ).normalize()


def superposition_results(
    sequent1: Sequent,
    sequent2: Sequent
) -> list[Sequent]:
    """Produces a list of all sequents which can be obtained by applying superposition to the given sequents.

    Parameters
    ----------
    toplevel: Sequent
        The Sequent from which the toplevel term instance is obtained.
    other: Sequent
        The Sequent from which the general term instance is obtained.

    Returns
    -------
    list[Sequent]
        The list of sequents obtained by superposition.
    """
    results = []
    for toplevel_terminstance in positive_toplevel_terminstances(sequent1):
        for terminstance in terminstances(sequent2):
            result = superposition(toplevel_terminstance, terminstance)
            if result is not None:
                results.append((toplevel_terminstance, terminstance, result))
    for toplevel_terminstance in positive_toplevel_terminstances(sequent2):
        for terminstance in terminstances(sequent1):
            result = superposition(toplevel_terminstance, terminstance)
            if result is not None:
                results.append((toplevel_terminstance, terminstance, result))
    return results


class Proof:
    """A proof.
    """

    def __init__(
            self,
            problem: Problem,
            derived_sequents: list[Sequent],
            derivation: list[Union[int, tuple[int,
                                              int, TermInstance, TermInstance]]]
    ):
        """Initializes a proof.
        """
        self.problem = problem
        self.derived_sequents = derived_sequents
        self.derivation = derivation

    def __str__(self):
        result = ""
        for i, sequent in enumerate(self.derived_sequents):
            if type(self.derivation[i]) == int:
                result += "{}: {}\t[Axiom]".format(i, sequent)
            else:
                if sequent in self.problem.conjectures:
                    result += "Goal\n{}: {}\t[Superposition {} {}]\n".format(
                        i, sequent, self.derivation[i][0], self.derivation[i][1]
                    )
                else:
                    result += "{}: {}\t[Superposition {} {}\n]".format(
                        i, sequent, self.derivation[i][0], self.derivation[i][1]
                    )
        return result

    def check(self):
        """Checks if the proof is valid."""
        index = self.problem.axioms + self.derived_sequents
        for i, sequent in enumerate(self.derived_sequents):
            if index[self.derivation[i][0]] != self.derivation[i][2].sequent:
                raise InvalidProof(
                    "Step {}: Term instance {} ({}) does not match sequent {} ({}).".format(
                        i, self.derivation[i][0], sequent, self.derivation[i][0], self.derivation[i][2].sequent
                    )
                )
            if index[self.derivation[i][1]] != self.derivation[i][3].sequent:
                raise InvalidProof(
                    "Step {}: Term instance {} ({}) does not match sequent {} ({}).".format(
                        i, self.derivation[i][1], sequent, self.derivation[i][1], self.derivation[i][3].sequent
                    )
                )
            if len(self.problem.axioms) + i < self.derivation[i][0] or len(self.problem.axioms) + i < self.derivation[i][1]:
                raise InvalidProof(
                    "Step {}: Attempting to use sequent {} that has not been derived yet.".format(
                        i, max(self.derivation[i][0], self.derivation[i][1]))
                )
            if self.derived_sequents[i] != superposition(self.derivation[i][2], self.derivation[i][3]):
                raise InvalidProof(
                    "Step {}: Derived formula ({}) does not match superposition of {} ({}) and {} ({}).".format(
                        i,
                        self.derived_sequents[i],
                        self.derivation[i][0],
                        self.derivation[i][2].sequent,
                        self.derivation[i][1],
                        self.derivation[i][3].sequent
                    )
                )
        for conjecture in self.problem.conjectures:
            if conjecture not in self.derived_sequents:
                raise InvalidProof(
                    "Conjecture {} is not derived.".format(conjecture))


class InvalidProof(Exception):
    """An error in the proof.
    """

    def __init__(self, message: str):
        """Initializes the error.
        """
        self.message = message

    def __str__(self):
        return self.message
