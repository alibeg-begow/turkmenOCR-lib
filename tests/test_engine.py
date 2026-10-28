from PIL import Image

from turkmenocr.engine import MAX_IMAGE_SIDE, PROMPTS, RESOLUTIONS, clean_output, prepare_image


def test_prompts_contain_image_placeholder():
    assert all(prompt.startswith("<image>\n") for prompt in PROMPTS.values())


def test_default_resolution_tiles_pages():
    assert RESOLUTIONS["gundam"].crop_mode
    assert not RESOLUTIONS["base"].crop_mode


def test_clean_output_strips_layout_tags_and_special_tokens():
    raw = (
        "<|ref|>title<|/ref|><|det|>[[10, 20, 30, 40]]<|/det|>\n# Salam dünýä\n\n\n\n"
        "Türkmenistan<｜end▁of▁sentence｜>"
    )
    assert clean_output(raw) == "# Salam dünýä\n\nTürkmenistan"


def test_clean_output_keeps_table_markup():
    assert clean_output("<table><tr><td>ň</td></tr></table>") == "<table><tr><td>ň</td></tr></table>"


def test_transparent_image_gets_white_background():
    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    result = prepare_image(rgba)
    assert result.mode == "RGB"
    assert result.getpixel((5, 5)) == (255, 255, 255)


def test_huge_image_is_downscaled_proportionally(tmp_path):
    path = tmp_path / "big.png"
    Image.new("L", (MAX_IMAGE_SIDE * 2, MAX_IMAGE_SIDE), 255).save(path)
    result = prepare_image(path)
    assert result.size == (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE // 2)
