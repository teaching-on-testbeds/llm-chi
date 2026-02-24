SINGLE_GPU ?= a100

INTRO_SRC := snippets/intro.md
INTRO_NOTEBOOK := 0_intro.ipynb
SINGLE_INDEX := single/index_$(SINGLE_GPU).md
INDEX_A100 := index_a100.md
INDEX_H100 := index_h100.md
MULTI_INDEX := multi/index.md
ADVANCE_RESERVE_A100 := advance_reserve_a100.md
ADVANCE_RESERVE_H100 := advance_reserve_h100.md
ADVANCE_RESERVE_SRC := single/snippets/intro.md single/snippets/reserve.md multi/snippets/intro.md multi/snippets/reserve.md
GPU_FILTER := single/filters/gpu_select.lua

.PHONY: all index advance_reserve single multi a100 h100 clean

all: index $(INDEX_A100) $(INDEX_H100) $(ADVANCE_RESERVE_A100) $(ADVANCE_RESERVE_H100) $(INTRO_NOTEBOOK)

index: index.md

advance_reserve: $(ADVANCE_RESERVE_A100) $(ADVANCE_RESERVE_H100)

single:
	$(MAKE) -C single $(SINGLE_GPU)

multi:
	$(MAKE) -C multi index.md

single/index_%.md:
	$(MAKE) -C single $*

$(MULTI_INDEX):
	$(MAKE) -C multi index.md

index.md: $(INTRO_SRC) $(SINGLE_INDEX) $(MULTI_INDEX)
	cat $(INTRO_SRC) $(SINGLE_INDEX) $(MULTI_INDEX) > index.md

$(INDEX_A100): $(INTRO_SRC) single/index_a100.md $(MULTI_INDEX)
	cat $(INTRO_SRC) single/index_a100.md $(MULTI_INDEX) > $(INDEX_A100)

$(INDEX_H100): $(INTRO_SRC) single/index_h100.md $(MULTI_INDEX)
	cat $(INTRO_SRC) single/index_h100.md $(MULTI_INDEX) > $(INDEX_H100)

$(ADVANCE_RESERVE_A100): $(ADVANCE_RESERVE_SRC) $(GPU_FILTER)
	cat $(ADVANCE_RESERVE_SRC) > advance_reserve.a100.tmp.md
	GPU=a100 pandoc --standalone --wrap=none --lua-filter $(GPU_FILTER) --from markdown --to markdown \
		-o advance_reserve.a100.filtered.md advance_reserve.a100.tmp.md
	grep -v '^:::' advance_reserve.a100.filtered.md > $(ADVANCE_RESERVE_A100)
	rm advance_reserve.a100.tmp.md advance_reserve.a100.filtered.md

$(ADVANCE_RESERVE_H100): $(ADVANCE_RESERVE_SRC) $(GPU_FILTER)
	cat $(ADVANCE_RESERVE_SRC) > advance_reserve.h100.tmp.md
	GPU=h100 pandoc --standalone --wrap=none --lua-filter $(GPU_FILTER) --from markdown --to markdown \
		-o advance_reserve.h100.filtered.md advance_reserve.h100.tmp.md
	grep -v '^:::' advance_reserve.h100.filtered.md > $(ADVANCE_RESERVE_H100)
	rm advance_reserve.h100.tmp.md advance_reserve.h100.filtered.md

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
	rm -f index.md index_a100.md index_h100.md advance_reserve.md advance_reserve_a100.md advance_reserve_h100.md 0_intro.ipynb
	$(MAKE) -C single clean
	$(MAKE) -C multi clean
