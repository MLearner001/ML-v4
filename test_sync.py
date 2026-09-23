import pandas as pd
from trace_synchronizer import SinuTrainSynchronizer

def test_sync():
    # Mock data
    gcode_data = {
        'N_Number': [1, 2, 3, 4, -1, 5],
        'Is_G01': [0, 1, 1, 1, 1, 0],
        'Is_G02': [0, 0, 0, 0, 0, 0],
        'Is_G03': [0, 0, 0, 0, 0, 0],
        'Is_Cycle800': [0, 0, 0, 0, 0, 0],
        'Cmd_F': [0, 5000, 1000, 2000, 2000, 0],
        'Delta_3D': [0.0, 10.0, 0.05, 0.05, 0.05, 0.0],
        'Delta_Rot': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    df_gcode = pd.DataFrame(gcode_data)

    trace_data = {
        'mapped_block': ['2', '2', '2', '5'],
        'f7\\s7': [4900, 5000, 5100, 0]
    }
    df_trace = pd.DataFrame(trace_data)

    syncer = SinuTrainSynchronizer()
    result = syncer.match_and_calculate_targets(df_gcode, df_trace)

    print("Result df:\n", result[['N_Number', 'Cmd_F', 'Target_Feedrate', 'Duration_Sec']])

    # Validation logic
    assert (result['Target_Feedrate'] <= 20000.0).all(), "Feedrate exceeded physical machine limit!"

    # Make sure we don't exceed Cmd_F if it is > 0
    mask_cmd = result['Cmd_F'] > 0
    assert (result.loc[mask_cmd, 'Target_Feedrate'] <= result.loc[mask_cmd, 'Cmd_F'] + 1e-4).all(), "Target_Feedrate exceeded Cmd_F!"

    # Check durations for small movement
    mask_movement = result['Delta_3D'] > 1e-4
    assert (result.loc[mask_movement, 'Duration_Sec'] > 0.0).all(), "Duration too close to zero!"

    print("All tests passed.")

if __name__ == "__main__":
    test_sync()
