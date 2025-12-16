# Target *.tex file without the extension
TARGET    ?= beamer
BUILD_DIR ?= build

TARGET_FILES = $(wildcard *.tex) \
               $(wildcard *.sty) \
			   $(wildcard res/*.svg) \
			   $(wildcard res/*.png) \
			   $(wildcard res/*.jpg) \
			   $(wildcard res/*.pdf)

$(TARGET).pdf: $(TARGET_FILES)
	@mkdir -p $(BUILD_DIR)
	@printf '\def\\builddir{$(BUILD_DIR)}' >builddir.def
	pdflatex -interaction nonstopmode -shell-escape -output-directory $(BUILD_DIR) $(TARGET).tex
	pdflatex -shell-escape -output-directory $(BUILD_DIR) $(TARGET).tex
	@mv $(BUILD_DIR)/$(TARGET).pdf .

.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)
	@rm -rf builddir.def
.PHONY: clean-all
clean-all: clean
	@rm -f $(TARGET).pdf
