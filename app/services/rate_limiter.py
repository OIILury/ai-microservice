# app/services/rate_limiter.py
"""
Limiteur de débit basé sur slowapi, appliqué aux endpoints qui appellent le LLM
(/api/corriger, /api/trouver-page). Protège un device aux ressources limitées (Jetson)
contre un usage abusif (bot, boucle de script) venant du site web public qui consomme
cette API.

Calibrage : usage normal attendu de l'ordre de 2-3 échanges par jour et par utilisateur.
Une limite de 10 requêtes/minute par IP laisse une large marge pour un usage légitime
(y compris quelqu'un qui teste plusieurs fois de suite) tout en bloquant un abus
automatisé. À ajuster si l'usage réel s'avère différent.

Note pour plus tard : si un reverse proxy (nginx ou autre) est ajouté devant ce service
(ex. pour le HTTPS), request.client.host renverra alors systématiquement l'IP du proxy
plutôt que celle du client réel, ce qui rendrait ce rate limiting par IP inopérant (tous
les clients comptés comme une seule IP). Il faudra à ce moment-là lire l'en-tête
X-Forwarded-For (en s'assurant que le proxy le renseigne et que rien d'autre que le
proxy ne peut l'usurper). Pour l'instant, sans proxy, l'IP de connexion brute est fiable.
"""
from slowapi import Limiter
from starlette.requests import Request


def get_client_ip(request: Request) -> str:
    """
    Détermine l'IP du client à utiliser comme clé de rate limiting.
    Accès direct (pas de reverse proxy pour l'instant) : l'IP de connexion brute
    correspond bien à l'IP réelle du client.
    """
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=get_client_ip)