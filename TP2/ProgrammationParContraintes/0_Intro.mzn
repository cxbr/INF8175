include "globals.mzn";

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Nombre devant satisfaire les critères
var 1000..9999: nombre;

% Décomposition du nombre en un ensemble de chiffres
array[1..4] of var 1..9: chiffres;

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Variable à minimiser
var int: p;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
constraint nombre = chiffres[1] * 1000 + chiffres[2] * 100 + chiffres[3] * 10 + chiffres[4];
constraint nombre mod 2 = 0;
constraint count(chiffres, 0) = 0;
constraint all_different(chiffres);
constraint chiffres[1] > chiffres[2];
constraint chiffres[3] < chiffres[4];
constraint chiffres[2] > chiffres[4];
constraint (chiffres[1] + chiffres[2] + chiffres[3] + chiffres[4]) > 15;
constraint (chiffres[2] * chiffres[3] * chiffres[4]) = p;

solve minimize p;
%-----------------------------------------------------------------------------%

output [
  show(nombre)
];

