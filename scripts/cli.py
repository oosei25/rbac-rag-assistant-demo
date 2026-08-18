import argparse

from app.services.indexer import indexer_service
from app.services.rag import rag_service


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "ask"])
    ap.add_argument("--role", default="employee")
    ap.add_argument("--q", help="question")
    args = ap.parse_args()

    if args.cmd == "ingest":
        n = indexer_service.reindex()
        print(f"Indexed chunks: {n}")
    else:
        answer, citations = rag_service.generate(
            args.q or "What do we know about company events?", args.role
        )
        print(answer)
        if citations:
            print("\nCitations:")
            for citation in citations:
                print(
                    f"[{citation.citation_id}] {citation.title} "
                    f"({citation.department}) - {citation.path}"
                )


if __name__ == "__main__":
    main()
