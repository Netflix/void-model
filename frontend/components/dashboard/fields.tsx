export function TextField({
  value,
  onChange,
  placeholder,
  label,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  label?: string;
}) {
  return (
    <label className="block text-xs text-zinc-400">
      {label ?? placeholder}
      <input
        className="mt-1 w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </label>
  );
}

export function NumberField({
  value,
  onChange,
  placeholder,
  label,
}: {
  value: number;
  onChange: (value: number) => void;
  placeholder: string;
  label?: string;
}) {
  return (
    <label className="block text-xs text-zinc-400">
      {label ?? placeholder}
      <input
        className="mt-1 w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
        type="number"
        value={String(value)}
        onChange={(e) => onChange(Number(e.target.value))}
        placeholder={placeholder}
      />
    </label>
  );
}
