GPU ?= a100

FILTER := filters/gpu_select.lua
INDEX_OUT := index.md
INDEX_A100_OUT := index_a100.md
INDEX_H100_OUT := index_h100.md
ADVANCE_RESERVE_OUT := advance_reserve.md
INTRO_OUT := 0_intro.ipynb
RESERVE_OUT := 1_reserve.ipynb
CREATE_OUT := 2_create_server.ipynb
SINGLE_WORKSPACE_OUT := workspace/2_single_gpu.ipynb

.PHONY: all build a100 h100 clean validate-gpu

all: build

build: validate-gpu $(INDEX_A100_OUT) $(INDEX_H100_OUT) $(INDEX_OUT) $(ADVANCE_RESERVE_OUT) $(INTRO_OUT) $(RESERVE_OUT) $(CREATE_OUT) $(SINGLE_WORKSPACE_OUT)

a100:
	$(MAKE) build GPU=a100

h100:
	$(MAKE) build GPU=h100

validate-gpu:
	@if [ "$(GPU)" != "a100" ] && [ "$(GPU)" != "h100" ]; then \
		echo "Unsupported GPU '$(GPU)'. Use GPU=a100 or GPU=h100."; \
		exit 1; \
	fi

$(INDEX_A100_OUT): snippets/intro.md snippets/reserve.md snippets/create_server.md snippets/single_gpu.md snippets/footer.md $(FILTER)
	cat snippets/intro.md \
		snippets/reserve.md \
		snippets/create_server.md \
		snippets/single_gpu.md \
		> index.a100.tmp.md
	GPU=a100 pandoc --standalone --wrap=none --lua-filter $(FILTER) --from markdown --to markdown \
		-o index.a100.filtered.md index.a100.tmp.md
	grep -v '^:::' index.a100.filtered.md > $(INDEX_A100_OUT)
	rm index.a100.tmp.md index.a100.filtered.md
	cat snippets/footer.md >> $(INDEX_A100_OUT)

$(INDEX_H100_OUT): snippets/intro.md snippets/reserve.md snippets/create_server.md snippets/single_gpu.md snippets/footer.md $(FILTER)
	cat snippets/intro.md \
		snippets/reserve.md \
		snippets/create_server.md \
		snippets/single_gpu.md \
		> index.h100.tmp.md
	GPU=h100 pandoc --standalone --wrap=none --lua-filter $(FILTER) --from markdown --to markdown \
		-o index.h100.filtered.md index.h100.tmp.md
	grep -v '^:::' index.h100.filtered.md > $(INDEX_H100_OUT)
	rm index.h100.tmp.md index.h100.filtered.md
	cat snippets/footer.md >> $(INDEX_H100_OUT)

$(INDEX_OUT): $(INDEX_A100_OUT) $(INDEX_H100_OUT)
	cp index_$(GPU).md $(INDEX_OUT)

$(ADVANCE_RESERVE_OUT): snippets/intro.md snippets/reserve.md $(FILTER)
	cat snippets/intro.md snippets/reserve.md > advance_reserve.tmp.md
	GPU=$(GPU) pandoc --standalone --wrap=none --lua-filter $(FILTER) --from markdown --to markdown \
		-o advance_reserve.filtered.md advance_reserve.tmp.md
	grep -v '^:::' advance_reserve.filtered.md > $(ADVANCE_RESERVE_OUT)
	rm advance_reserve.tmp.md advance_reserve.filtered.md

$(INTRO_OUT): snippets/frontmatter_python.md snippets/intro.md $(FILTER)
	GPU=$(GPU) pandoc --resource-path=../ --embed-resources --standalone --wrap=none --lua-filter $(FILTER) \
		-i snippets/frontmatter_python.md snippets/intro.md \
		-o $(INTRO_OUT)
	sed -i 's/attachment://g' $(INTRO_OUT)

$(RESERVE_OUT): snippets/frontmatter_python.md snippets/reserve.md $(FILTER)
	GPU=$(GPU) pandoc --resource-path=../ --embed-resources --standalone --wrap=none --lua-filter $(FILTER) \
		-i snippets/frontmatter_python.md snippets/reserve.md \
		-o $(RESERVE_OUT)
	sed -i 's/attachment://g' $(RESERVE_OUT)

$(CREATE_OUT): snippets/frontmatter_python.md snippets/create_server.md $(FILTER)
	GPU=$(GPU) pandoc --resource-path=../ --embed-resources --standalone --wrap=none --lua-filter $(FILTER) \
		-i snippets/frontmatter_python.md snippets/create_server.md \
		-o $(CREATE_OUT)
	sed -i 's/attachment://g' $(CREATE_OUT)

$(SINGLE_WORKSPACE_OUT): snippets/frontmatter_bash.md snippets/single_gpu.md $(FILTER)
	GPU=$(GPU) pandoc --resource-path=../ --embed-resources --standalone --wrap=none --lua-filter $(FILTER) \
		-i snippets/frontmatter_bash.md snippets/single_gpu.md \
		-o $(SINGLE_WORKSPACE_OUT)
	sed -i 's/attachment://g' $(SINGLE_WORKSPACE_OUT)

clean:
	rm -f index.md index_a100.md index_h100.md advance_reserve.md 0_intro.ipynb 1_reserve.ipynb 2_create_server.ipynb workspace/2_single_gpu.ipynb
