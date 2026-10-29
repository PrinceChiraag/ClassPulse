"""Compatibility entry point for deployments configured to import ``app:app``.

The authenticated portal is implemented in :mod:`classpulse.backend`. Using it
here keeps Docker and platform deployments on the same login flow as ``wsgi``.
"""

from classpulse.backend import create_app

app = create_app()


def get_start_url():
    """Return the teacher sign-in page used by local browser launchers."""
    return "http://localhost:5000/login"


if __name__ == "__main__":
    import os

    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 10000)))
