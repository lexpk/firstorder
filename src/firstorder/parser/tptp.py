from functools import reduce
from lark import Lark, Token, Tree
from importlib.resources import files
from ..logic import (
    Equation,
    Function,
    Problem,
    Sequent,
    Term,
    Variable,
    substitute
)


tptp_grammar = files('firstorder.parser').joinpath(
    'tptp_grammar.lark').read_text()
tptp_parser = Lark(tptp_grammar, start='tptp_file')


def from_tptp_file(
    file: str,
    include_dir: str = ""
) -> Problem:
    axioms, hypotheses, functions = _from_tptp_file(file, include_dir)
    return _from_axioms_hypotheses_functions(axioms, hypotheses, functions)


def _from_tptp_file(
        file: str,
        include_dir: str = ""
) -> tuple[list[Sequent], list[Sequent], set[tuple[str, int, bool]]]:
    with open(file, 'r') as f:
        input = parse_tptp_tree(
            tptp_parser.parse(f.read()))
    axioms = input["axiom"]
    hypotheses = input["conjecture"]
    functions = input["functions"]
    for include in input["include"]:
        result = _from_tptp_file(include_dir + include)
        axioms += result[0]
        hypotheses += result[1]
        functions.update(result[2])
    return axioms, hypotheses, functions


def from_tptp_string(str: str) -> Problem:
    tree = tptp_parser.parse(str)
    input = parse_tptp_tree(tree)
    axioms = input["axiom"]
    hypotheses = input["conjecture"]
    functions = input["functions"]
    return _from_axioms_hypotheses_functions(axioms, hypotheses, functions)


def _from_axioms_hypotheses_functions(axioms, hypotheses, functions) -> Problem:
    skolemized_axioms = []
    skolemized_hypotheses = []
    skolem_counter = 0
    for axiom in axioms:
        skolemized_axiom, skolem_functions = skolemize(axiom, skolem_counter)
        skolemized_axioms.append(skolemized_axiom)
        functions.update(skolem_functions)
        skolem_counter += len(skolem_functions)
    for hypothesis in hypotheses:
        skolemized_hypothesis, skolem_functions = skolemize(
            hypothesis,
            skolem_counter
        )
        skolemized_hypotheses.append(skolemized_hypothesis)
        functions.update(skolem_functions)
        skolem_counter += len(skolem_functions)
    final_axioms = []
    final_hypotheses = []
    for axiom in skolemized_axioms:
        final_axioms += cnf(axiom)
    for hypothesis in skolemized_hypotheses:
        final_hypotheses += cnf(hypothesis)
    problem = Problem()
    problem.declare_function("true", 0)
    problem.declare_function("false", 0)
    problem.declare_function("type", 1)
    problem.declare_function("bool", 0)
    problem.read_axiom("true = false ->")
    problem.read_axiom("type(?X) = bool -> ?X = true, ?X = false")
    problem.read_axiom("-> type(true) = bool")
    problem.read_axiom("-> type(false) = bool")
    for function in functions:
        if function[0] not in problem.functions:
            problem.declare_function(
                function[0],
                function[1]
            )
    for axiom in final_axioms:
        problem.read_axiom(repr(axiom))
    for hypothesis in final_hypotheses:
        problem.read_conjecture(repr(hypothesis))
    return problem


def to_tptp_file(p: Problem, file: str):
    with open(file, 'w') as f:
        f.write(to_tptp_string(p))


def to_tptp_string(p: Problem) -> str:
    result = ""
    for i, axiom in enumerate(p.axioms):
        result += f"cnf(axiom{i}, axiom, {to_tptp_string_sequent(axiom)}).\n"
    for i, conjecture in enumerate(p.conjectures):
        result += f"fof(conjecture{i}, conjecture, {to_tptp_string_sequent(conjecture)}).\n"
    return result


def to_tptp_string_sequent(s: Sequent) -> str:
    result = ""
    if not s.left and not s.right:
        result += "false"
    else:
        for i, equation in enumerate(s.left):
            if i > 0:
                result += " | "
            result += repr(equation).replace("=", "!=")
        if s.left and s.right:
            result += " | "
        for i, equation in enumerate(s.right):
            if i > 0:
                result += " | "
            result += repr(equation)
    return result


def parse_tptp_tree(input: Tree) -> dict:
    result = {
        "include": [],
        "axiom": [],
        "conjecture": [],
        "functions": set(),
        "variables": {},
    }
    _parse_tptp_tree(input, result)
    return result


def _parse_tptp_tree(input: Tree, result: dict):
    for i in input.children:
        match i.children[0].data.value:
            case "include":
                result["include"].append(
                    i.children[0].children[0].value.strip("'")
                )
            case "annotated_formula":
                _parse_annotated_formula(i.children[0], result)


def _parse_annotated_formula(input: Tree, data: dict):
    match input.children[0].data.value:
        case "fof_annotated":
            return _parse_fof_annotated(input.children[0], data)
        case "cnf_annotated":
            return _parse_cnf_annotated(input.children[0], data)
        case _:
            raise ValueError(f"Only 'fof_annotated' or 'cnf_annotated' allowed, \
input contains {input.children[0].data.value}")


def _parse_fof_annotated(input: Tree, data: dict):
    match input.children[1].value:
        case "axiom" | "hypothesis" | "definition" | "assumption" | "lemma" | "theorem":
            data["axiom"].append(
                _parse_fof_formula(input.children[2], data)
            )
        case "conjecture":
            data["conjecture"].append(
                _parse_fof_formula(input.children[2], data)
            )
        case "negated_conjecture":
            data["conjecture"].append(
                Function("not", (_parse_fof_formula(input.children[2], data),))
            )


def _parse_cnf_annotated(input: Tree, data: dict):
    match input.children[1].value:
        case "axiom" | "hypothesis" | "definition" | "assumption" | "lemma" | "theorem":
            data["axiom"].append(
                _parse_disjunction(input.children[2].children[0], data)
            )
        case "conjecture":
            data["conjecture"].append(
                _parse_disjunction(input.children[2].children[0], data)
            )
        case "negated_conjecture":
            data["conjecture"].append(
                Function("not", (_parse_disjunction(
                    input.children[2].children[0], data),))
            )


def _parse_disjunction(input: Tree, data: dict) -> Term:
    if len(input.children) == 1:
        return _parse_literal(input.children[0], data)
    else:
        return reduce(lambda x, y: Function("or", (x, y)),
                      map(lambda z: _parse_literal(z, data), input.children))


def _parse_literal(input: Tree, data: dict) -> Term:
    match input.children[0].data.value:
        case "fof_atomic_formula":
            return _parse_fof_atomic_formula(input.children[0], data)
        case "negated_fof_atomic_formula":
            return Function("not", (_parse_fof_atomic_formula(
                input.children[0].children[0], data),))
        case "fof_infix_unary":
            return _parse_fof_infix_unary(input.children[0], data)


def _parse_fof_formula(input: Tree, data: dict) -> Term:
    match input.children[0].data.value:
        case "fof_logic_formula":
            return _parse_fof_logic_formula(input.children[0], data)
        case _:
            raise ValueError(
                f"Only 'fof_logic_formula'allowed, \
input contains {input.children[0].data.value}")


def _parse_fof_logic_formula(input: Tree, data: dict) -> list[Term]:
    match input.children[0].data.value:
        case "fof_binary_formula":
            return _parse_fof_binary_formula(input.children[0], data)
        case "fof_unitary_formula":
            return _parse_fof_unitary_formula(input.children[0], data)
        case _:
            raise ValueError(f"Only 'fof_binary_formula', \
'fof_unitary_formula' allowed, input contains {input.children[0].data.value}")


def _parse_fof_binary_formula(input: Tree, data: dict) -> list[Term]:
    match input.children[0].data.value:
        case "fof_binary_nonassoc":
            return _parse_fof_binary_nonassoc(input.children[0], data)
        case "fof_binary_assoc":
            return _parse_fof_binary_assoc(input.children[0], data)
        case _:
            raise ValueError(f"Only 'fof_binary_nonassoc' and \
'fof_binary_assoc' allowed, input contains {input.data.value}")


def _parse_fof_binary_nonassoc(input: Tree, data: dict) -> Term:
    left = _parse_fof_unitary_formula(input.children[0], data)
    op = input.children[1].value
    right = _parse_fof_unitary_formula(input.children[2], data)
    match op:
        case "<=>":
            return Function("iff", (left, right))
        case "=>":
            return Function("implies", (left, right))
        case "<=":
            return Function("implies", (right, left))
        case "<~>":
            return Function("xor", (left, right))
        case "~|":
            return Function("nor", (left, right))
        case "~&":
            return Function("nand", (left, right))


def _parse_fof_binary_assoc(input: Tree, data: dict) -> Term:
    match input.children[0].data.value:
        case "fof_or_formula":
            return _parse_fof_or_formula(input.children[0], data)
        case "fof_and_formula":
            return _parse_fof_and_formula(input.children[0], data)
        case _:
            raise ValueError(f"Only 'fof_or_formula' and \
'fof_and_formula' allowed, input contains {input.data.value}")


def _parse_fof_or_formula(input: Tree, data: dict) -> Term:
    match input.children[0].data.value:
        case "fof_unitary_formula":
            left = _parse_fof_unitary_formula(input.children[0], data)
        case "fof_or_formula":
            left = _parse_fof_or_formula(input.children[0], data)
    right = _parse_fof_unitary_formula(input.children[1], data)
    return Function("or", (left, right))


def _parse_fof_and_formula(input: Tree, data: dict) -> Term:
    match input.children[0].data.value:
        case "fof_unitary_formula":
            left = _parse_fof_unitary_formula(input.children[0], data)
        case "fof_and_formula":
            left = _parse_fof_and_formula(input.children[0], data)
    right = _parse_fof_unitary_formula(input.children[1], data)
    return Function("and", (left, right))


def _parse_fof_unitary_formula(input: Tree, data: dict) -> Term:
    match input.children[0].data.value:
        case "fof_quantified_formula":
            return _parse_fof_quantified_formula(input.children[0], data)
        case "fof_unary_formula":
            return _parse_fof_unary_formula(input.children[0], data)
        case "fof_atomic_formula":
            return _parse_fof_atomic_formula(input.children[0], data)
        case "fof_logic_formula":
            return _parse_fof_logic_formula(input.children[0], data)


def _parse_fof_quantified_formula(input: Tree, data: dict) -> Term:
    quantifier = input.children[0].value
    variables = _parse_fof_variable_list(input.children[1], data)
    formula = _parse_fof_unitary_formula(input.children[3], data)
    match quantifier:
        case "!":
            return Function(f"forall{len(variables)}", variables + (formula,))
        case "?":
            return Function(f"exists{len(variables)}", variables + (formula,))


def _parse_fof_variable_list(input: Tree, data: dict) -> list[Term]:
    return tuple(
        _parse_variable(i, data) for i in input.children
    )


def _parse_variable(input: Tree, data: dict) -> Term:
    if input not in data["variables"]:
        data["variables"][input] = Variable(len(data["variables"]))
    return data["variables"][input]


def _parse_fof_unary_formula(input: Tree, data: dict) -> Term:
    match input.children[0].data.value:
        case "fof_unitary_formula":
            return Function(
                "not",
                (_parse_fof_unitary_formula(input.children[0], data),)
            )
        case "fof_infix_unary":
            return _parse_fof_infix_unary(input.children[0], data)


def _parse_fof_infix_unary(input: Tree, data: dict) -> Term:
    left = _parse_fof_term(input.children[0], data)
    right = _parse_fof_term(input.children[2], data)
    return Function("neq", (left, right))


def _parse_fof_term(input: Tree, data: dict) -> Term:
    if type(input.children[0]) == Token:
        return _parse_variable(input.children[0], data)
    match input.children[0].data.value:
        case "fof_function_term":
            return _parse_fof_function_term(input.children[0], data)
        case _:
            raise ValueError(f"Only 'fof_function_term' and 'VARIABLE' \
                allowed, input contains {input.children[0].data.value}")


def _parse_fof_function_term(input: Tree, data: dict) -> Term:
    match input.children[0].data.value:
        case "fof_plain_term":
            return _parse_fof_plain_term(input.children[0], data, False)
        case _:
            raise ValueError(f"Only 'fof_plain_term' \
                allowed, input contains {input.children[0].data.value}")


def _parse_fof_plain_term(
        input: Tree | Token,
        data: dict,
        predicate: bool
) -> Term:
    data["functions"].add(
        (input.children[0].value,
         len(input.children), predicate)
    )
    match len(input.children):
        case 1:
            return Function(input.children[0].value, tuple())
        case _:
            return Function(
                input.children[0].value,
                tuple(
                    _parse_fof_term(i, data)
                    for i in input.children[1].children
                )
            )


def _parse_fof_plain_atomic_formula(input: Tree, data: dict) -> Term:
    return _parse_fof_plain_term(input.children[0], data, True)


def _parse_fof_atomic_formula(input: Tree, data: dict) -> Term:
    match input.children[0].data.value:
        case "fof_plain_atomic_formula":
            return _parse_fof_plain_atomic_formula(input.children[0], data)
        case "fof_defined_atomic_formula":
            return _parse_fof_defined_atomic_formula(input.children[0], data)
        case _:
            raise ValueError(f"Only 'fof_plain_atomic_formula' \
allowed, input contains {input.children[0].data.value}")


def _parse_fof_defined_atomic_formula(input: Tree, data: dict) -> Term:
    match input.children[0].data.value:
        case "fof_defined_infix_formula":
            return _parse_fof_defined_infix_formula(input.children[0], data)
        case "fof_defined_plain_formula":
            return _parse_fof_defined_plain_formula(input.children[0], data)


def _parse_fof_defined_plain_formula(input: Tree, data: dict) -> Term:
    data["functions"].add((input.children[0].value.replace("$", ""), 0, True))
    return Function(input.children[0].value.replace("$", ""), tuple())


def _parse_fof_defined_infix_formula(input: Tree, data: dict) -> Term:
    left = _parse_fof_term(input.children[0], data)
    right = _parse_fof_term(input.children[2], data)
    return Function("eq", (left, right))


def skolemize(
    term: Term,
    skolem_counter: int


) -> tuple[Term, set[str, int, bool]]:
    skolem_functions = set()
    return (
        _skolemize(
            term,
            {"counter": skolem_counter},
            skolem_functions,
            set(),
            True
        ),
        skolem_functions
    )


def _skolemize(
    term: Term,
    skolem_counter: dict,
    skolem_functions: dict,
    free_variables: set[Variable],
    arity: bool
) -> Term:
    if type(term) == Variable:
        return term
    if type(term) == Function:
        if term.name.startswith("forall"):
            match arity:
                case True:
                    return _skolemize(
                        term.args[-1],
                        skolem_counter,
                        skolem_functions,
                        free_variables | set(term.args[:-1]),
                        True
                    )
                case False:
                    substition = {
                        term.args[i]: Function(
                            f"sk{skolem_counter['counter'] + i}",
                            tuple(free_variables)
                        )
                        for i in range(len(term.args) - 1)
                    }
                    skolem_functions.update(
                        [
                            (
                                f"sk{skolem_counter['counter'] + i}",
                                len(free_variables),
                                False
                            ) for i in range(len(term.args) - 1)
                        ]
                    )
                    skolem_counter["counter"] += len(term.args) - 1
                    return _skolemize(
                        substitute(substition, term.args[-1]),
                        skolem_counter,
                        skolem_functions,
                        free_variables,
                        False
                    )
        if term.name.startswith("exists"):
            match arity:
                case True:
                    substition = {
                        term.args[i]: Function(
                            f"sk{skolem_counter['counter'] + i}",
                            tuple(free_variables)
                        )
                        for i in range(len(term.args) - 1)
                    }
                    skolem_functions.update(
                        [
                            (
                                f"sk{skolem_counter['counter'] + i}",
                                len(free_variables),
                                False
                            ) for i in range(len(term.args) - 1)
                        ]
                    )
                    skolem_counter["counter"] += len(term.args) - 1
                    return _skolemize(
                        substitute(substition, term.args[-1]),
                        skolem_counter,
                        skolem_functions,
                        free_variables,
                        True
                    )
                case False:
                    return _skolemize(
                        term.args[-1],
                        skolem_counter,
                        skolem_functions,
                        free_variables | set(term.args[:-1]),
                        False
                    )
        match term.name:
            case "not":
                return Function(
                    "not",
                    (
                        _skolemize(
                            term.args[0],
                            skolem_counter,
                            skolem_functions,
                            free_variables,
                            not arity
                        ),
                    )
                )
            case "and":
                return Function(
                    "and",
                    tuple(
                        _skolemize(
                            child,
                            skolem_counter,
                            skolem_functions,
                            free_variables,
                            arity
                        )
                        for child in term.args
                    )
                )
            case "or":
                return Function(
                    "or",
                    tuple(
                        _skolemize(
                            child,
                            skolem_counter,
                            skolem_functions,
                            free_variables,
                            arity
                        )
                        for child in term.args
                    )
                )
            case "implies":
                return _skolemize(
                    Function(
                        "or",
                        (
                            Function("not", (term.args[0],)),
                            term.args[1],
                        )
                    ),
                    skolem_counter,
                    skolem_functions,
                    free_variables,
                    arity
                )
            case "iff":
                return _skolemize(
                    Function(
                        "and",
                        (
                            Function("implies", (term.args[0], term.args[1],)),
                            Function("implies", (term.args[1], term.args[0],)),
                        )
                    ),
                    skolem_counter,
                    skolem_functions,
                    free_variables,
                    arity
                )
            case "xor":
                return _skolemize(
                    Function(
                        "or",
                        (
                            Function(
                                "and",
                                (Function(
                                    "not", (term.args[0],)), term.args[1])
                            ),
                            Function(
                                "and",
                                (term.args[0], Function(
                                    "not", (term.args[1],)))
                            ),
                        )
                    ),
                    skolem_counter,
                    skolem_functions,
                    free_variables,
                    arity
                )
            case "nor":
                return _skolemize(
                    Function("not", (Function("or", term.args),)),
                    skolem_counter,
                    skolem_functions,
                    free_variables,
                    arity
                )
            case "nand":
                return _skolemize(
                    Function("not", (Function("and", term.args),)),
                    skolem_counter,
                    skolem_functions,
                    free_variables,
                    arity
                )
            case _:
                return term


def cnf(term: Term) -> list[Sequent]:
    sequents = []
    for clause in _cnf(term):
        sequents.append(
            Sequent(
                tuple(x for x, y in clause if not y),
                tuple(x for x, y in clause if y)
            )
        )
    return sequents


def _cnf(term: Term) -> list[list[tuple[Term, Term]]]:
    """
    returns a term in conjunctive normal form

    >>> cnf(Function("and", (Function("or", (Variable(1), Variable(2))),
        Function("or", (Variable(3), Variable(4))))))
    [[(Variable(1), True), (Variable(2), True)],
    [(Variable(3), True), (Variable(4), True)]]
    """
    match term.name:
        case "not":
            d = _dnf(term.args[0])
            return [[(x, not y) for x, y in clause] for clause in d]
        case "and":
            return _cnf(term.args[0]) + _cnf(term.args[1])
        case "or":
            c1 = _cnf(term.args[0])
            c2 = _cnf(term.args[1])
            return [x + y for x in c1 for y in c2]
        case "eq":
            return [[(Equation(term.args[0], term.args[1]), True)]]
        case "neq":
            return [[(Equation(term.args[0], term.args[1]), False)]]
        case _:
            return [[(Equation(term, Function("true", tuple())), True)]]


def dnf(term: Term) -> list[Sequent]:
    sequents = []
    for clause in _dnf(term):
        sequents.append(
            Sequent(
                tuple(x for x, y in clause if not y),
                tuple(x for x, y in clause if y)
            )
        )
    return sequents


def _dnf(term: Term) -> list[list[tuple[Term, Term]]]:
    match term.name:
        case "not":
            c = _cnf(term.args[0])
            return [[(x, not y) for x, y in clause] for clause in c]
        case "or":
            return _dnf(term.args[0]) + _dnf(term.args[1])
        case "and":
            d1 = _dnf(term.args[0])
            d2 = _dnf(term.args[1])
            return [x + y for x in d1 for y in d2]
        case "eq":
            return [[(Equation(term.args[0], term.args[1]), True)]]
        case "neq":
            return [[(Equation(term.args[0], term.args[1]), False)]]
        case _:
            return [[(Equation(term, Function("true", tuple())), True)]]
