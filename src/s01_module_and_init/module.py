from __future__ import annotations


from src.s00_autograd_and_tensors.autograd import BasicTensor

class BasicParameter(BasicTensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class BasicModule:

    def __init__(self):
        self._children = {}
        self._parameters = {}
        self.training = True

    def add_child(self, name: str, child: BasicModule):
        """
        Ajoute ou remplace un module de nom "name" sous-module au module actuel
        """
        self._children[name] = child

    def add_param(self, name: str, param: BasicParameter):
        """
        Ajoute ou remplace un param de nom "name" au module actuel
        """
        self._parameters[name] = param

    def get_param(self, name: str):
        """
        Récupère le paramètre de nom "name" dans les paramètres du module courant
        """
        try:
            return self._parameters[name]
        except KeyError:
            raise KeyError(f"Le paramètre de nom '{name}' n'existe pas dans les paramètres de ce module.") from None

    def get_child(self, name: str):
        """
        Récupère le sous-module de nom "name" dans les enfants du module actuel
        """
        try:
            return self._children[name]
        except KeyError:
            raise KeyError(f"Le module de nom '{name}' n'existe pas dans les sous-modules de ce module.") from None

    def get_submodule(self, name: str):
        """
        Renvoie le sous-module de nom "name".
        Permet de chercher récursivement un module de format "sous-module.sous-sous-module.[...].module", sans
        récupérer les sous-modules des sous-modules
        """
        splited = name.split(".")
        current = self
        path = []
        try:
            for elem in splited:
                path.append(elem)
                current = current.get_child(elem)

            return current
        except KeyError:
            raise KeyError(f"Le module de nom '{name}' n'existe pas dans les sous-modules de ce module. "
                           f"Echec au niveau de : {".".join(path)} ") from None

    def get_subparam(self, name: str):
        """
        Renvoie le paramètre de nom "name".
        Permet de chercher récursivement un module de format "sous-module.sous-sous-module.[...].param", sans
        récupérer les sous-modules.
        """
        splited = name.split(".")
        if len(splited) == 1:
            module_path = None
            last = self
        else:
            module_path = ".".join(splited[:-1])

            last = self.get_submodule(module_path)

        try:
            return last.get_param(splited[-1])
        except KeyError:
            if module_path is not None:
                raise KeyError(f"Le module {module_path} existe dans ce module "
                           f"mais ne contient pas de paramètre '{splited[-1]}'") from None
            else:
                raise KeyError(f"Le paramètre '{splited[-1]}' n'existe pas dans le module") from None

    def children(self, recursive=False, _seen=None):
        """
        Renvoie un itérateur sur les sous-modules du module courant.
        Si recursive est à "True", renvoie tous les sous-modules, récursivement.
        Un module enregistré à plusieurs endroits n'est renvoyé qu'une fois.
        """
        if not recursive:
            yield from self._children.values()
            return

        if _seen is None:
            _seen = set()

        for child in self._children.values():
            if child not in _seen:
                _seen.add(child)
                yield child
                yield from child.children(recursive=True, _seen=_seen)

    def parameters(self, recursive=True):
        """
        Renvoie un itérateur sur les paramètres du module courant.
        Si recursive est à "True", renvoie aussi les paramètres des sous-modules, récursivement.
        Ne renvoie qu'une fois un même paramètre, même s'il est utilisé à deux endroits différents
        """
        modules = [self]
        if recursive:
            modules.extend(self.children(recursive=True))

        seen = set()
        for module in modules:
            for param in module._parameters.values():
                if param not in seen:
                    seen.add(param)
                    yield param


    def apply(self, fn):
        """
        Applique une même fonction 'fn' récursivement à ce module et ses sous-modules

        fn doit être une lambda de forme fn(Module) -> None
        """
        fn(self)
        for module in self.children(recursive=True):
            fn(module)
        return self

    def train(self, mode=True):
        """Passe le module et tous ses descendants en mode entraînement (ou éval si mode=False)."""
        self.apply(lambda m: setattr(m, "training", mode))
        return self

    def eval(self):
        """Equivalent à train(False)."""
        return self.train(False)

    def zero_grad(self):
        for param in self.parameters(recursive=True):
            param.zero_grad()
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward n'est pas défini pour ce module.")

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if hasattr(self, "_parameters") and hasattr(self, "_children"):
            self._parameters.pop(name, None)
            self._children.pop(name, None)

            if isinstance(value, BasicModule):
                self.add_child(name, value)
            if isinstance(value, BasicParameter):
                self.add_param(name, value)

        else:
            if isinstance(value, BasicModule):
                raise AttributeError("Ajout d'un sous-module avant d'avoir initialisé les dictionnaires. Veuillez faire "
                                 "'super().__init__()' avant tout ajout de module")
            if isinstance(value, BasicParameter):
                raise AttributeError("Ajout d'un paramètre avant d'avoir initialisé les dictionnaires. Veuillez faire "
                                 "'super().__init__()' avant tout ajout de paramètre")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


