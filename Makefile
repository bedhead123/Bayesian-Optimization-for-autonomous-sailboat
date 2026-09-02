.PHONY: run dry-run quick-test hyper-test setup clean
# One-command entry — `make` == dry-run
run: dry-run
dry-run:
	./run.sh --dry-run
quick-test:
	./run.sh --quick-test
hyper-test:
	./run.sh --hyper-test
setup:
	./run.sh --dry-run
clean:
	rm -rf output/*.db output/*.log output/quick_test output/hyper_test output/medium_test
