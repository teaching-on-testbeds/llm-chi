SINGLE_GPU ?= a100

INTRO_SRC := snippets/intro.md
INTRO_NOTEBOOK := 0_intro.ipynb
SINGLE_INDEX := single/index_$(SINGLE_GPU).md
MULTI_INDEX := multi/index.md
ADVANCE_RESERVE := advance_reserve.md

.PHONY: all index advance_reserve single multi a100 h100 clean

all: index $(ADVANCE_RESERVE) $(INTRO_NOTEBOOK)

index: index.md

advance_reserve: $(ADVANCE_RESERVE)

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

$(ADVANCE_RESERVE): single/snippets/intro.md single/snippets/reserve.md multi/snippets/intro.md multi/snippets/reserve.md
	cat single/snippets/intro.md single/snippets/reserve.md multi/snippets/intro.md multi/snippets/reserve.md > advance_reserve.tmp.md
	grep -v '^:::' advance_reserve.tmp.md > $(ADVANCE_RESERVE)
	rm advance_reserve.tmp.md

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
	rm -f index.md advance_reserve.md 0_intro.ipynb
	$(MAKE) -C single clean
	$(MAKE) -C multi clean
