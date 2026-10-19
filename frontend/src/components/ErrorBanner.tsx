export function ErrorBanner({
  title = "Error",
  messages,
}: {
  title?: string;
  messages: string | string[];
}) {
  const list = Array.isArray(messages) ? messages : [messages];
  if (list.length === 0) return null;
  return (
    <div className="error-banner" role="alert">
      <strong>{title}</strong>
      {list.length === 1 ? (
        <span> — {list[0]}</span>
      ) : (
        <ul>
          {list.map((m, i) => (
            <li key={i}>{m}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
