#!/usr/bin/env bash

# Consente di avviare la pagina anche da un'altra cartella.
cd -- "$(dirname -- "$0")" || exit 1

pausa_errore() {
    if [[ -t 0 ]]; then
        read -r -p "Premi Invio per chiudere... "
    fi
}

uv_command=$(command -v uv)
if [[ -z "$uv_command" && -x "$HOME/.local/bin/uv" ]]; then
    uv_command="$HOME/.local/bin/uv"
fi
if [[ -z "$uv_command" ]]; then
    printf '%s\n' "Impossibile avviare la pagina: manca uv." \
        "Chiedi a chi ha preparato il progetto di completare l'installazione."
    pausa_errore
    exit 127
fi

printf '%s\n' "Avvio della pagina: il browser si aprirà automaticamente." \
    "Al primo avvio potrebbe servire qualche minuto." \
    "Lascia aperta questa finestra. Per chiudere la pagina, premi Ctrl+C."
trap 'printf "\nPagina chiusa.\n"; exit 130' INT
"$uv_command" run main.py step=page
status=$?
if [[ "$status" -ne 0 && "$status" -ne 130 && "$status" -ne 143 ]]; then
    printf '\n%s\n' "La pagina non si è avviata correttamente." \
        "Mostra i messaggi qui sopra a chi ha preparato il progetto."
    pausa_errore
fi
exit "$status"
