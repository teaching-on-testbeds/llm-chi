GPU ?= a100

FILTER := ../single/filters/gpu_select.lua
INDEX_OUT := index.md
INTRO_OUT := 0_intro.ipynb
RESERVE_OUT := 1_reserve.ipynb
CREATE_OUT := 2_create_server.ipynb
WORKSPACE_OUT := workspace/3_multi_gpu.ipynb

.PHONY: all a100 h100 clean validate-gpu

all: $(INDEX_OUT) $(INTRO_OUT) $(RESERVE_OUT) $(CREATE_OUT) $(WORKSPACE_OUT)

a100:
	$(MAKE) all GPU=a100

h100:
	$(MAKE) all GPU=h100

validate-gpu:
	@if [ "$(GPU)" != "a100" ] && [ "$(GPU)" != "h100" ]; then \
		echo "Unsupported GPU '$(GPU)'. Use GPU=a100 or GPU=h100."; \
		exit 1; \
	fi

$(INDEX_OUT): snippets/*.md $(FILTER) validate-gpu
	cat snippets/intro.md \
		snippets/reserve.md \
		snippets/create_server.md \
		snippets/multi_gpu.md \
		> index.tmp.md
	GPU=$(GPU) pandoc --standalone --wrap=none --lua-filter $(FILTER) --from markdown --to markdown \
		-o index.filtered.md index.tmp.md
	grep -v '^:::' index.filtered.md > $(INDEX_OUT)
	rm index.tmp.md index.filtered.md
	cat snippets/footer.md >> $(INDEX_OUT)

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
