"""Optional instalation of CmdStan (requires the extra ``stan``)."""

from __future__ import annotations

import sys
from typing import Optional


def install_stan(directory: Optional[str] = None, overwrite: bool = False) -> str:
    """Instala o CmdStan e retorna o caminho da instalação.

    Parameters
    ----------
    directory:
        Pasta de destino. Se ``None``, usa o padrão do cmdstanpy (``~/.cmdstan``).
    overwrite:
        Reinstala mesmo que a versão já exista.
    """
    try:
        import cmdstanpy
    except ImportError:
        raise ImportError(
            "cmdstanpy is not installed. Run in your terminal:\n"
            '    pip install "mteresa[stan]"'
        ) from None

    print("Installing CmdStan (this can take 2-10 minutes)...")
    ok = cmdstanpy.install_cmdstan(
        dir=directory,
        overwrite=overwrite,
        compiler=(sys.platform == "win32"),
    )
    if not ok:
        raise RuntimeError("CmdStan installation failed; see the output above.")

    path = str(cmdstanpy.cmdstan_path())
    print(f"CmdStan installed at: {path}")
    return path


if __name__ == "__main__":
    install_stan(sys.argv[1] if len(sys.argv) > 1 else None)