all: index.md 0_intro.ipynb 1_create_server.ipynb 2_single_gpu_a100.ipynb 3_multi_gpu_a100.ipynb 3_multi_gpu_v100.ipynb

clean: 
	rm index.md 0_intro.ipynb 1_create_server.ipynb 2_single_gpu_a100.ipynb workspace/2_single_gpu_a100.ipynb 3_multi_gpu_a100.ipynb 3_multi_gpu_v100.ipynb

index.md: snippets/*.md 
	cat snippets/intro.md \
		snippets/create_server.md \
		snippets/single_gpu_a100.md \
		snippets/multi_gpu_a100.md \
		snippets/multi_gpu_v100.md \
		> index.tmp.md
	grep -v '^:::' index.tmp.md > index.md
	rm index.tmp.md
	cat snippets/footer.md >> index.md

0_intro.ipynb: snippets/intro.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/intro.md \
                -o 0_intro.ipynb  
	sed -i 's/attachment://g' 0_intro.ipynb


1_create_server.ipynb: snippets/create_server.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/create_server.md \
                -o 1_create_server.ipynb  
	sed -i 's/attachment://g' 1_create_server.ipynb

2_single_gpu_a100.ipynb: snippets/single_gpu_a100.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/single_gpu_a100.md \
                -o 2_single_gpu_a100.ipynb  
	sed -i 's/attachment://g' 2_single_gpu_a100.ipynb
	cp 2_single_gpu_a100.ipynb workspace/2_single_gpu_a100.ipynb
	
3_multi_gpu_a100.ipynb: snippets/multi_gpu_a100.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/multi_gpu_a100.md \
                -o 3_multi_gpu_a100.ipynb  
	sed -i 's/attachment://g' 3_multi_gpu_a100.ipynb

3_multi_gpu_v100.ipynb: snippets/multi_gpu_v100.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/multi_gpu_v100.md \
                -o 3_multi_gpu_v100.ipynb  
	sed -i 's/attachment://g' 3_multi_gpu_v100.ipynb



