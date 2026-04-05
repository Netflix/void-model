export function TextField({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
}) {
  return (
    <input
      className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
    />
  );
}

export function NumberField({
  value,
  onChange,
  placeholder,
}: {
  value: number;
  onChange: (value: number) => void;
  placeholder: string;
}) {
  return (
    <input
      className="w-full rounded-md border border-zinc-700 px-3 py-2 text-sm"
      type="number"
      value={String(value)}
      onChange={(e) => onChange(Number(e.target.value))}
      placeholder={placeholder}
    />
  );
}
