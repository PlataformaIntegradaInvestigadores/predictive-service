"""Errores del dominio. Los adaptadores (HTTP) deciden cómo mostrarlos."""


class AffiliationNotFound(LookupError):
    """La afiliación pedida no existe o no tiene historial de publicaciones."""


class GroupNotFound(KeyError):
    """No existe un grupo persistente con ese identificador."""
