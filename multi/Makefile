GPU ?= a100

FILTER := ../single/filters/gpu_select.lua
INDEX_OUT := index.md
INDEX_A100_OUT := index_a100.md
INDEX_H100_OUT := index_h100.md
INTRO_OUT := 0_intro.ipynb
RESERVE_OUT := 1_reserve.ipynb
CREATE_OUT := 2_create_server.ipynb
WORKSPACE_OUT := workspace/3_multi_gpu.ipynb

.PHONY: all build a100 h100 clean validate-gpu

all: build

build: validate-gpu $(INDEX_A100_OUT) $(INDEX_H100_OUT) $(INDEX_OUT) $(INTRO_OUT) $(RESERVE_OUT) $(CREATE_OUT) $(WORKSPACE_OUT)

a100:
	$(MAKE) build GPU=a100

h100:
	$(MAKE) build GPU=h100

validate-gpu:
	@if [ "$(GPU)" != "a100" ] && [ "$(GPU)" != "h100" ]; then \
		echo "Unsupported GPU '$(GPU)'. Use GPU=a100 or GPU=h100."; \
		exit 1; \
	fi

$(INDEX_A100_OUT): snippets/*.md $(FILTER)
	cat snippets/intro.md \
		snippets/reserve.md \
		snippets/create_server.md \
		snippets/multi_gpu.md \
		> index.a100.tmp.md
	GPU=a100 pandoc --standalone --wrap=none --lua-filter $(FILTER) --from markdown --to markdown \
		-o index.a100.filtered.md index.a100.tmp.md
	grep -v '^:::' index.a100.filtered.md > $(INDEX_A100_OUT)
	rm index.a100.tmp.md index.a100.filtered.md
	cat snippets/footer.md >> $(INDEX_A100_OUT)

$(INDEX_H100_OUT): snippets/*.md $(FILTER)
	cat snippets/intro.md \
		snippets/reserve.md \
		snippets/create_server.md \
		snippets/multi_gpu.md \
		> index.h100.tmp.md
	GPU=h100 pandoc --standalone --wrap=none --lua-filter $(FILTER) --from markdown --to markdown \
		-o index.h100.filtered.md index.h100.tmp.md
	grep -v '^:::' index.h100.filtered.md > $(INDEX_H100_OUT)
	rm index.h100.tmp.md index.h100.filtered.md
	cat snippets/footer.md >> $(INDEX_H100_OUT)

$(INDEX_OUT): $(INDEX_A100_OUT) $(INDEX_H100_OUT)
	cp index_$(GPU).md $(INDEX_OUT)

$(INTRO_OUT): snippets/frontmatter_python.md snippets/intro.md $(FILTER) validate-gpu
	GPU=$(GPU) pandoc --resource-path=../ --embed-resources --standalone --wrap=none --lua-filter $(FILTER) \
		-i snippets/frontmatter_python.md snippets/intro.md \
		-o $(INTRO_OUT)
	sed -i 's/attachment://g' $(INTRO_OUT)

$(RESERVE_OUT): snippets/frontmatter_python.md snippets/reserve.md $(FILTER) validate-gpu
	GPU=$(GPU) pandoc --resource-path=../ --embed-resources --standalone --wrap=none --lua-filter $(FILTER) \
		-i snippets/frontmatter_python.md snippets/reserve.md \
		-o $(RESERVE_OUT)
	sed -i 's/attachment://g' $(RESERVE_OUT)

$(CREATE_OUT): snippets/frontmatter_python.md snippets/create_server.md $(FILTER) validate-gpu
	GPU=$(GPU) pandoc --resource-path=../ --embed-resources --standalone --wrap=none --lua-filter $(FILTER) \
		-i snippets/frontmatter_python.md snippets/create_server.md \
		-o $(CREATE_OUT)
	sed -i 's/attachment://g' $(CREATE_OUT)

$(WORKSPACE_OUT): snippets/frontmatter_python.md snippets/multi_gpu.md $(FILTER) validate-gpu
	GPU=$(GPU) pandoc --resource-path=../ --embed-resources --standalone --wrap=none --lua-filter $(FILTER) \
		-i snippets/frontmatter_python.md snippets/multi_gpu.md \
		-o $(WORKSPACE_OUT)
	sed -i 's/attachment://g' $(WORKSPACE_OUT)

clean:
	rm -f $(INDEX_OUT) $(INTRO_OUT) $(RESERVE_OUT) $(CREATE_OUT) $(WORKSPACE_OUT)
