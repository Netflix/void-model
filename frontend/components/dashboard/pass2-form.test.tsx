import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Pass2FormPanel } from "@/components/dashboard/pass2-form";

describe("Pass2FormPanel", () => {
  it("invokes cache action callbacks", () => {
    const onCheckCache = vi.fn();
    const onClearCache = vi.fn();

    render(
      <Pass2FormPanel
        pass2VideoNames="lime"
        setPass2VideoNames={vi.fn()}
        pass2DataRoot="./sample"
        setPass2DataRoot={vi.fn()}
        pass2Pass1Dir="./outputs"
        setPass2Pass1Dir={vi.fn()}
        pass2OutputDir="./out"
        setPass2OutputDir={vi.fn()}
        pass2ModelName="./model"
        setPass2ModelName={vi.fn()}
        pass2ModelCheckpoint="./void_pass2.safetensors"
        setPass2ModelCheckpoint={vi.fn()}
        pass2Height="384"
        setPass2Height={vi.fn()}
        pass2Width="672"
        setPass2Width={vi.fn()}
        pass2GuidanceScale="6.0"
        setPass2GuidanceScale={vi.fn()}
        pass2Steps="50"
        setPass2Steps={vi.fn()}
        pass2WarpedNoiseCacheDir="./cache"
        setPass2WarpedNoiseCacheDir={vi.fn()}
        pass2SkipNoiseGeneration={false}
        setPass2SkipNoiseGeneration={vi.fn()}
        pass2UseQuadmask
        setPass2UseQuadmask={vi.fn()}
        cachePath="./cache"
        setCachePath={vi.fn()}
        cacheInfo={null}
        onCheckCache={onCheckCache}
        onClearCache={onClearCache}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Check" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));

    expect(onCheckCache).toHaveBeenCalledTimes(1);
    expect(onClearCache).toHaveBeenCalledTimes(1);
  });
});
