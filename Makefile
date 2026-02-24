SINGLE_GPU ?= a100

INTRO_SRC := snippets/intro.md
INTRO_NOTEBOOK := 0_intro.ipynb
SINGLE_INDEX := single/index_$(SINGLE_GPU).md
MULTI_INDEX := multi/index.md

.PHONY: all index single multi a100 h100 clean

all: index $(INTRO_NOTEBOOK)

index: index.md

single:
	$(MAKE) -C single $(SINGLE_GPU)

multi:
	$(MAKE) -C multi index.md

$(SINGLE_INDEX):
	$(MAKE) -C single $(SINGLE_GPU)

$(MULTI_INDEX):
	$(MAKE) -C multi index.md

index.md: $(INTRO_SRC) $(SINGLE_INDEX) $(MULTI_INDEX)
	cat $(INTRO_SRC) $(SINGLE_INDEX) $(MULTI_INDEX) > index.md

$(INTRO_NOTEBOOK): $(INTRO_SRC) snippets/frontmatter_python.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
		-i snippets/frontmatter_python.md $(INTRO_SRC) \
		-o $(INTRO_NOTEBOOK)
	sed -i 's/attachment://g' $(INTRO_NOTEBOOK)

a100:
	$(MAKE) index SINGLE_GPU=a100

h100:
	$(MAKE) index SINGLE_GPU=h100

clean:
	rm -f index.md 0_intro.ipynb
	$(MAKE) -C single clean
	$(MAKE) -C multi clean
