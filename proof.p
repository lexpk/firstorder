% Running in auto input_syntax mode. Trying TPTP
% Refutation found. Thanks to Tanya!
% SZS status Theorem for transformed
% SZS output start Proof for transformed
fof(r17, conjecture, true != true | true=subset(b, bb)). % superposition
fof(pr10, axiom, (! [X0, X1]: (true != equal_sets(X0, X1) | true=subset(X0, X1)))).
fof(pr5, axiom, true=equal_sets(b, bb)).
%
fof(r18, conjecture, true=subset(b, bb)). % trivial inequality removal
fof(pr17, axiom, true != true | true=subset(b, bb)).
%
fof(r21, conjecture, (! [X0]: (true != true | true=member(X0, bb) | true != member(X0, b)))). % superposition
fof(pr7, axiom, (! [X2, X0, X1]: (true != subset(X1, X2) | true=member(X0, X2) | true != member(X0, X1)))).
fof(pr18, axiom, true=subset(b, bb)).
%
fof(r24, conjecture, (! [X0]: (true != member(X0, b) | true=member(X0, bb)))). % trivial inequality removal
fof(pr21, axiom, (! [X0]: (true != true | true=member(X0, bb) | true != member(X0, b)))).
%
fof(r34, conjecture, true != true | true=member(element_of_b, bb)). % superposition
fof(pr24, axiom, (! [X0]: (true != member(X0, b) | true=member(X0, bb)))).
fof(pr6, axiom, true=member(element_of_b, b)).
%
fof(r35, conjecture, true=member(element_of_b, bb)). % trivial inequality removal
fof(pr34, axiom, true != true | true=member(element_of_b, bb)).
%
fof(r36, conjecture, $false). % subsumption resolution
fof(pr35, axiom, true=member(element_of_b, bb)).
fof(pr16, axiom, true != member(element_of_b, bb)).
%
%
% SZS output end Proof for transformed
% ------------------------------
% Version: Vampire 4.7 (commit 2d02e4655 on 2022-07-11 21: 15: 24 + 0200)
% Linked with Z3 4.8.13.0 f03d756e086f81f2596157241e0decfb1c982299 z3-4.8.4-5390-gf03d756e0
% Termination reason: Refutation

% Memory used[KB]: 4989
% Time elapsed: 0.034 s
% ------------------------------
% ------------------------------
