# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACION (modifique SOLO este bloque)
# ============================================================
DIRECTORY = '/home/user/Septiembre_2025/RFM_Vortex/RFM_one_segment_250nm.out'

MAG_COMPONENT   = 1   # 1 = mx, 2 = my, 3 = mz  (para MAGNITUD)
PHASE_COMPONENT = 3   # 1 = mx, 2 = my, 3 = mz  (para FASE)
# COLS_SPEC = "61,111,262,394"  # Ejemplo: "3", "3,7,10-15", "1-20,50,100"  ("" para no reducir)
COLS_SPEC = "4,58,137,235,295,346"

# Carpetas de salida (se crean si no existen)
OUTPUT_DIR_MAG   = os.path.join(DIRECTORY, 'Output_Modes')
OUTPUT_DIR_PHASE = os.path.join(DIRECTORY, 'Output_Phase')
os.makedirs(OUTPUT_DIR_MAG, exist_ok=True)
os.makedirs(OUTPUT_DIR_PHASE, exist_ok=True)
# ============================================================


def parse_spec(spec: str):
    """Convierte cadenas tipo '3,7,10-15' en lista de enteros [3,7,10,11,12,13,14,15]."""
    cols = []
    spec = (spec or "").strip()
    if not spec:
        return cols
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            a, b = token.split('-', 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            cols.extend(range(a, b + 1))
        else:
            cols.append(int(token))
    return cols


def read_odt_component(directory, files, col_index):
    """
    Lee la columna especifica (mx/my/mz) por celda y tiempo.
    Filtra las filas donde m = (0,0,0) (valores nulos).
    Devuelve un DataFrame con filas = celdas y columnas = tiempos.
    """
    columns = []
    for filename in files:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as f:
            # saltar las primeras 12 lineas de cabecera
            for _ in range(12):
                next(f)
            # filtrar filas validas
            filtered = [
                line for line in f
                if len(line.split()) >= 6 and not (
                    float(line.split()[3]) == 0.0 and
                    float(line.split()[4]) == 0.0 and
                    float(line.split()[5]) == 0.0
                )
            ]
            # extraer valores de la columna deseada
            col_vals = [float(line.split()[col_index]) for line in filtered]
            columns.append(col_vals)
    # reorganizar: filas = celdas, columnas = tiempos
    return pd.DataFrame(list(map(list, zip(*columns))), dtype=float)


def get_positions_df(rep_file):
    """
    Extrae posiciones pos_x, pos_y, pos_z de un archivo .odt representativo.
    Usa el mismo filtrado de m != (0,0,0).
    """
    with open(rep_file, 'r') as f:
        for _ in range(12):
            next(f)
        data_pos = [
            line.split()[:6] for line in f
            if len(line.split()) >= 6 and not (
                float(line.split()[3]) == 0.0 and
                float(line.split()[4]) == 0.0 and
                float(line.split()[5]) == 0.0
            )
        ]
    df_pos = pd.DataFrame(
        data_pos,
        columns=['pos_x', 'pos_y', 'pos_z', 'col4', 'col5', 'col6']
    ).apply(pd.to_numeric)
    return df_pos[['pos_x', 'pos_y', 'pos_z']].reset_index(drop=True)


def run_fft_per_cell_keep_dc_zero(df_time):
    """
    Calcula la FFT para cada celda (fila) a lo largo del tiempo (columnas).
    La componente DC se fija en cero (fft[:,0]=0).
    Devuelve:
      - magnitud FFT por celda
      - fase FFT por celda
      - indice de corte (half_index) para frecuencias positivas
    """
    data_vals = df_time.values
    fft_result = np.fft.fft(data_vals, axis=1)
    fft_result[:, 0] = 0.0 + 0.0j  # DC = 0

    half_index = fft_result.shape[1] // 2 + 1  # incluye Nyquist
    fft_pos = fft_result[:, :half_index]
    abs_fft = np.abs(fft_pos)
    phase_fft = np.angle(fft_pos)
    freq_cols = [str(i) for i in range(1, half_index + 1)]
    return (pd.DataFrame(abs_fft, columns=freq_cols),
            pd.DataFrame(phase_fft, columns=freq_cols),
            half_index)


def reduce_columns(df_full, logical_cols, allowed_max):
    """
    Mantiene pos_x,pos_y,pos_z y las columnas de frecuencia seleccionadas.
    Devuelve DataFrame reducido.
    """
    if not logical_cols:
        return None  # No hay reduccion solicitada

    bad = [c for c in logical_cols if c < 1 or c > allowed_max]
    if bad:
        raise ValueError(f"Columnas fuera de rango (1..{allowed_max}): {bad}")

    # Mapeo: pos_x,pos_y,pos_z = columnas 0,1,2; luego frecuencias
    positions = [0, 1, 2] + [c + 2 for c in logical_cols]

    ncols = df_full.shape[1]
    if positions and max(positions) >= ncols:
        raise ValueError(f"La seleccion requiere hasta la columna {max(positions)+1}, "
                         f"pero el archivo tiene {ncols} columnas.")
    return df_full.iloc[:, positions]


def main():
    # Validacion de componentes
    if MAG_COMPONENT not in (1, 2, 3) or PHASE_COMPONENT not in (1, 2, 3):
        raise ValueError("MAG_COMPONENT y PHASE_COMPONENT deben ser 1 (mx), 2 (my) o 3 (mz).")

    # Mapear componentes a etiquetas y a indices de columna del .odt
    comp_map = {1: ('mx', 3), 2: ('my', 4), 3: ('mz', 5)}
    comp_mag_lbl, col_mag_idx = comp_map[MAG_COMPONENT]
    comp_phase_lbl, col_phase_idx = comp_map[PHASE_COMPONENT]

    # Listar archivos .odt ordenados por el numero temporal
    files = sorted(
        [f for f in os.listdir(DIRECTORY) if f.endswith('.odt')],
        key=lambda x: int(x.split('m')[1].split('.')[0])
    )
    if not files:
        raise FileNotFoundError("No se encontraron archivos .odt en la ruta indicada.")

    # Seleccionar archivo representativo para posiciones
    rep_file = os.path.join(DIRECTORY, 'm001000.odt')
    if not os.path.exists(rep_file):
        rep_file = os.path.join(DIRECTORY, files[-1])

    # Extraer posiciones
    df_pos = get_positions_df(rep_file)

    # ------------------------
    # MAGNITUD
    # ------------------------
    df_time_mag = read_odt_component(DIRECTORY, files, col_mag_idx)
    abs_fft_df, _, half_index_mag = run_fft_per_cell_keep_dc_zero(df_time_mag)
    final_mag_full = pd.concat([df_pos, abs_fft_df.reset_index(drop=True)], axis=1)

    mag_full_name = f"magnitud_{comp_mag_lbl}.csv"
    mag_full_path = os.path.join(OUTPUT_DIR_MAG, mag_full_name)
    final_mag_full.to_csv(mag_full_path, index=False)

    # ------------------------
    # FASE
    # ------------------------
    df_time_phase = read_odt_component(DIRECTORY, files, col_phase_idx)
    _, phase_fft_df, half_index_phase = run_fft_per_cell_keep_dc_zero(df_time_phase)
    final_phase_full = pd.concat([df_pos, phase_fft_df.reset_index(drop=True)], axis=1)

    phase_full_name = f"Fase_{comp_phase_lbl}.csv"
    phase_full_path = os.path.join(OUTPUT_DIR_PHASE, phase_full_name)
    final_phase_full.to_csv(phase_full_path, index=False)

    # ------------------------
    # REDUCCIONES (opcional, si COLS_SPEC no esta vacio)
    # ------------------------
    logical_cols = parse_spec(COLS_SPEC)
    allowed_max = min(half_index_mag, half_index_phase)

    mag_red_df = reduce_columns(final_mag_full, logical_cols, allowed_max)
    phase_red_df = reduce_columns(final_phase_full, logical_cols, allowed_max)

    if mag_red_df is not None:
        # --- Guardar versión sin normalizar ---
        mag_red_name_raw = f"reduce_magnitud_{comp_mag_lbl}_raw.csv"
        mag_red_path_raw = os.path.join(OUTPUT_DIR_MAG, mag_red_name_raw)
        mag_red_df.to_csv(mag_red_path_raw, index=False)

        # --- Crear y guardar versión normalizada ---
        mag_red_norm = mag_red_df.copy()
        cols_to_norm = mag_red_norm.columns[3:]  # ignorar pos_x,pos_y,pos_z
        mag_red_norm[cols_to_norm] = mag_red_norm[cols_to_norm].div(
            mag_red_norm[cols_to_norm].max()
        )

        mag_red_name = f"reduce_magnitud_{comp_mag_lbl}.csv"
        mag_red_path = os.path.join(OUTPUT_DIR_MAG, mag_red_name)
        mag_red_norm.to_csv(mag_red_path, index=False)

        print(f"- Magnitud (reducido sin normalizar): {mag_red_path_raw}  [{mag_red_df.shape[1]} columnas]")
        print(f"- Magnitud (reducido y normalizado): {mag_red_path}  [{mag_red_norm.shape[1]} columnas]")

    if phase_red_df is not None:
        phase_red_name = f"reduce_Fase_{comp_phase_lbl}.csv"
        phase_red_path = os.path.join(OUTPUT_DIR_PHASE, phase_red_name)
        phase_red_df.to_csv(phase_red_path, index=False)
        print(f"- Fase (reducido): {phase_red_path}  [{phase_red_df.shape[1]} columnas]")

    # ------------------------
    # RESUMEN
    # ------------------------
    print("Proceso completado.")
    print(f"- Magnitud (completo): {mag_full_path}  [{final_mag_full.shape[1]} columnas]")
    print(f"- Fase (completo): {phase_full_path}  [{final_phase_full.shape[1]} columnas]")


if __name__ == "__main__":
    main()
