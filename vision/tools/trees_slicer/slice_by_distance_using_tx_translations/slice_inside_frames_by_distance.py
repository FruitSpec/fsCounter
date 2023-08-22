import pandas as pd


def slice_inside_frames(path_slices_csv, path_translations_csv, frame_width, output_path=None):
    center_x_pixel = int(frame_width / 2)

    # Load data
    translations_df = pd.read_csv(path_translations_csv)
    slices_df = pd.read_csv(path_slices_csv, index_col=0)

    # todo - remove this, it's a bug fix
    min_len = min( len(slices_df), len(translations_df))
    slices_df = slices_df.iloc[:min_len]
    translations_df = translations_df.iloc[:min_len]
    ################################################################

    # Add 'frame' and 'tx' columns from translations dataframe to slices dataframe
    slices_df['frame_id'] = translations_df['frame'].values
    slices_df['tx'] = translations_df['tx'].values

    # Function to set start pixels for each group
    def set_start_pixels(group, slices_df, center_x_pixel=center_x_pixel):
        group = group.reset_index(drop=True)
        group.loc[group.index[0], 'start'] = center_x_pixel

        # Adjust 'start' based on 'tx' for each row in the group.
        # 'start' in the current row = 'start' in the previous row - 'tx' in the current row
        for i in range(1, len(group)):
            diff = group.loc[i - 1, 'start'] - group.loc[i, 'tx']
            diff = int(max(1, min(diff, frame_width)))
            group.loc[i, 'start'] = diff
            if diff == 1:
                break


        #################################################
        # add rows of previous tree_id group:
        duplicated_df = group.copy()
        duplicated_df = duplicated_df[duplicated_df['start'] > 1]
        duplicated_df['end'] = duplicated_df['start'] - 1
        duplicated_df['start'] = 1
        duplicated_df['tree_id'] = duplicated_df['tree_id'] - 1
        group = pd.concat([group, duplicated_df], ignore_index=True)

        ###################################################
        # Add rows to the group if 'start' for first row is not the end (less than frame_width)
        # a new row is added at the beginning of the group, with frame_id -1
        if group.loc[0, 'frame_id'] != 0:
            prev_tree_frames = slices_df.loc[slices_df['tree_id'] == group.loc[0, 'tree_id'] - 1] + len(duplicated_df)
            pervious_tree_last_frame=-1000
            if not prev_tree_frames.empty:
                pervious_tree_last_frame = prev_tree_frames.iloc[1]['frame_id']

            while (group.loc[0, 'start'] < frame_width) and (group.loc[0, 'start'] > 1):
                new_row = group.loc[0].copy()
                new_row['frame_id'] -= 1

                # Update 'tx' and 'start' for the new row
                new_row['tx'] = slices_df['tx'].loc[slices_df['frame_id'] == new_row['frame_id']]
                new_row['start'] += new_row['tx']
                if (int(new_row['start']) >= frame_width) or (int(new_row['start']) <= 1) :
                    break
                elif new_row['frame_id'] <= pervious_tree_last_frame:
                    break
                else:
                    group = pd.concat([pd.DataFrame(new_row).T, group]).reset_index(drop=True)

                    # Convert columns to integer type
                    for column in ['frame_id', 'tree_id', 'start', 'end']:
                        group[column] = group[column].astype(int)

        return group

    # Function to process start pixels for all groups
    def process_start_pixels(slices_df):
        slices_df = slices_df.groupby('tree_id').apply(set_start_pixels, slices_df=slices_df)
        # reset index:
        slices_df = slices_df.reset_index(drop=True)
        return slices_df

    # Function to set end pixels for each group
    def set_end_pixels(slices_df):

        slices_df = (slices_df.sort_values(by=['frame_id', 'tree_id'])).reset_index(drop=True)

        slices_df_copy = slices_df.copy()
        # Group the dataframe by 'frame_id'
        slices_df = slices_df.groupby('frame_id')

        # Update 'end' column for each group in slices_df copy
        for name, group in slices_df:
            for i in range(len(group) - 1):
                slices_df_copy.loc[group.index[i], 'end'] = slices_df_copy.loc[group.index[i + 1], 'start'] - 1

        # Regroup the data by 'frame_id' and concatenate back into a single dataframe
        grouped_by_frame_id = slices_df_copy.groupby('frame_id')
        updated_df = pd.concat([group for name, group in grouped_by_frame_id])

        return updated_df

    # process start pixels
    slices_df = process_start_pixels(slices_df)

    # Call the function to set end pixels and get the updated dataframe
    updated_df = set_end_pixels(slices_df)
    # updated_df = slices_df.copy()

    # Reset the index and remove the 'tx' column
    updated_df.reset_index(drop=True, inplace=True)
    #updated_df = updated_df.drop(columns=['tx'])

    # Save the updated dataframe to a csv file if output_path is provided
    if output_path is not None:
        updated_df.to_csv(output_path)
        print(f"Data saved to {output_path}")

    return updated_df

if __name__ == "__main__":
    '''
    This script adds pixels coordinates to slices.csv file that was generated automatically by distance
    '''
    updated_df = slice_inside_frames(path_slices_csv="/vision/trees_slicer/slice_by_distance_using_tx_translations/matan_dist/data_motcha_New_Hall_row_4/slices.csv",
                                     path_translations_csv="/vision/trees_slicer/slice_by_distance_using_tx_translations/matan_dist/data_motcha_New_Hall_row_4/jai_translations.csv",
                                     frame_width=1535,
                                     output_path="/vision/trees_slicer/slice_by_distance_using_tx_translations/matan_dist/data_motcha_New_Hall_row_4/all_slices.csv")


    print('Data processing done')

    """
    The first slice is -1, and it's just half of the first frame. 
    
    """

