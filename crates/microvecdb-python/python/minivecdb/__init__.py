"""
minivecdb — 1-bit quantised vector database.

The ``MiniVecDb`` class is a native Rust extension compiled via PyO3/Maturin.
For LangChain integration import from :mod:`minivecdb.langchain`::

    from minivecdb.langchain import LangChainMiniVecDb
"""

from minivecdb._minivecdb import MiniVecDb

__all__ = ["MiniVecDb"]
__version__ = "1.1.0"
