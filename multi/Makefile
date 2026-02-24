all: index.md advance_reserve.md 0_intro.ipynb 1_reserve.ipynb 2_create_server.ipynb workspace/3_multi_gpu.ipynb

clean: 
	rm index.md advance_reserve.md 0_intro.ipynb 1_reserve.ipynb 2_create_server.ipynb workspace/3_multi_gpu.ipynb

advance_reserve.md: snippets/intro.md snippets/reserve.md
	cat snippets/intro.md snippets/reserve.md > advance_reserve.tmp.md
	grep -v '^:::' advance_reserve.tmp.md > advance_reserve.md
	rm advance_reserve.tmp.md

index.md: snippets/*.md 
	cat snippets/intro.md \
		snippets/reserve.md \
		snippets/create_server.md \
		snippets/multi_gpu.md \
		> index.tmp.md
	grep -v '^:::' index.tmp.md > index.md
	rm index.tmp.md
	cat snippets/footer.md >> index.md

0_intro.ipynb: snippets/intro.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/intro.md \
                -o 0_intro.ipynb  
	sed -i 's/attachment://g' 0_intro.ipynb


1_reserve.ipynb: snippets/reserve.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/reserve.md \
                -o 1_reserve.ipynb  
	sed -i 's/attachment://g' 1_reserve.ipynb


2_create_server.ipynb: snippets/create_server.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/create_server.md \
		-o 2_create_server.ipynb  
	sed -i 's/attachment://g' 2_create_server.ipynb
	
workspace/3_multi_gpu.ipynb: snippets/multi_gpu.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
		-i snippets/frontmatter_python.md snippets/multi_gpu.md \
		-o workspace/3_multi_gpu.ipynb  
	sed -i 's/attachment://g' workspace/3_multi_gpu.ipynb
