from core._meta_helpers import apply_dye_wavelength_fallbacks, dye_wavelengths_from_name


def test_dye_lookup_recognizes_common_channel_names():
    assert dye_wavelengths_from_name("DAPI") == {
        "dye": "DAPI",
        "excitation_wavelength": 358.0,
        "emission_wavelength": 461.0,
    }
    assert dye_wavelengths_from_name("TRITC channel")["emission_wavelength"] == 576.0
    assert dye_wavelengths_from_name("Cy5")["excitation_wavelength"] == 650.0


def test_apply_dye_wavelength_fallbacks_uses_channel_names():
    meta = {
        "channel_names": ["DAPI", "TRITC", "Cy5"],
        "channels": [{}, {}, {}],
    }

    applied = apply_dye_wavelength_fallbacks(meta, 3)

    assert applied == {"emission_wavelength", "excitation_wavelength"}
    assert [ch["emission_wavelength"] for ch in meta["channels"]] == [461.0, 576.0, 670.0]
    assert [ch["excitation_wavelength"] for ch in meta["channels"]] == [358.0, 557.0, 650.0]
    assert meta["_inferred_keys"] == {"emission_wavelength", "excitation_wavelength"}
    assert len(meta["_dye_wavelength_fallbacks"]) == 3


def test_dye_fallback_does_not_overwrite_metadata_values():
    meta = {
        "channel_names": ["DAPI"],
        "channels": [{"emission_wavelength": 500.0}],
    }

    apply_dye_wavelength_fallbacks(meta, 1)

    assert meta["channels"][0]["emission_wavelength"] == 500.0
    assert meta["channels"][0]["excitation_wavelength"] == 358.0
    assert meta["channels"][0]["excitation_wavelength_source"] == "dye_name"
    assert "emission_wavelength_source" not in meta["channels"][0]
