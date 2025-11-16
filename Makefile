CC:=gcc
interpreter:=python

main: main.py libRION/libpso.so requirements.txt
	$(interpreter) $<

main.py: libRION/libRION.py
	@echo -e "Checking dependencies for $@"

libRION/libRION.py: libRION/libPSO.py libRION/libPSO_parallel.py
	@echo -e "Checking dependencies for $@"

libRION/libPSO_parallel.py: libRION/libPSO.c
	@echo -e "Checking dependencies for $@"

libRION/libpso.so: libRION/libPSO.c
	$(CC) -fPIC -fopenmp -O3 -o $@ $< -lm
