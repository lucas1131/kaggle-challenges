# Lucas Alexandre Soares Makefile - nusp 9293265

all: main

main: run

# Run directives
run:
	R -f src/titanic.r

.PHONY: list
list: 
	ls -lhR

.PHONY: tar
tar: 
	@echo Compressing files...
	@tar -zcvf titanic.tar.gz *
	@echo Done.

.PHONY: zip
zip: 
	@echo Compressing files...
	@zip -r titanic.zip *
	@echo Done.

git-show:
	git log --graph --full-history --all --pretty=format:"%h%x09%d%x20%s"
