cnf(axiom0, axiom, false != true).
cnf(axiom1, axiom, bool != type(X0) | false != true | X0 = true).
cnf(axiom2, axiom, bool = type(true)).
cnf(axiom3, axiom, bool = type(false)).
cnf(axiom4, axiom, equal_sets(b, bb) = true).
cnf(axiom5, axiom, member(element_of_b, b) = true).
cnf(axiom6, axiom, member(X0, X1) != true | subset(X1, X2) != true | member(X0, X2) = true).
cnf(axiom7, axiom, member(member_of_1_not_of_2(X0, X1), X0) != subset(X0, X1) | subset(X0, X1) = true).
cnf(axiom8, axiom, member(member_of_1_not_of_2(X0, X1), X1) != true | subset(X0, X1) = true).
cnf(axiom9, axiom, equal_sets(X0, X1) != true | subset(X0, X1) = true).
cnf(axiom10, axiom, equal_sets(X0, X1) != true | subset(X1, X0) = true).
cnf(axiom11, axiom, subset(X0, X1) != true | subset(X1, X0) != true | equal_sets(X1, X0) = true).
fof(conjecture0, conjecture, member(element_of_b, bb) = true).
