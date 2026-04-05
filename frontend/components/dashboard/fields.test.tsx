import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { NumberField, TextField } from "@/components/dashboard/fields";

describe("dashboard fields", () => {
  it("TextField emits updated string values", () => {
    const onChange = vi.fn();
    render(<TextField value="old" onChange={onChange} placeholder="config path" />);

    fireEvent.change(screen.getByPlaceholderText("config path"), { target: { value: "new-value" } });
    expect(onChange).toHaveBeenCalledWith("new-value");
  });

  it("NumberField emits updated numeric values", () => {
    const onChange = vi.fn();
    render(<NumberField value={10} onChange={onChange} placeholder="steps" />);

    fireEvent.change(screen.getByPlaceholderText("steps"), { target: { value: "42" } });
    expect(onChange).toHaveBeenCalledWith(42);
  });
});
