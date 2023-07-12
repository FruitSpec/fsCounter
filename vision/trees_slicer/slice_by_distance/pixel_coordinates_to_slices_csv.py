import pandas as pd


def process_slices_csv(path_slices_csv, path_translations_csv, frame_width, output_path=None):
    center_x_pixel = int(frame_width / 2)

    # Load data
    translations_df = pd.read_csv(path_translations_csv)
    slices_df = pd.read_csv(path_slices_csv, index_col=0)

    # Drop the last row of translations dataframe
    translations_df.drop(translations_df.index[-1], inplace=True)

    # Add 'frame' and 'tx' columns from translations dataframe to slices dataframe
    slices_df['frame_id'] = translations_df['frame'].values
    slices_df['tx'] = translations_df['tx'].values

    # Function to set start pixels for each group
    def set_start_pixels(group, slices_df, center_x_pixel=center_x_pixel):
        group = group.reset_index(drop=True)
        group.loc[group.index[0], 'start'] = center_x_pixel

        # Adjust 'start' based on 'tx' for each row in the group
        for i in range(1, len(group)):
            diff = group.loc[i - 1, 'start'] - group.loc[i, 'tx']
            diff = int(max(1, min(diff, frame_width)))
            group.loc[i, 'start'] = diff

        # Add rows to the group if 'start' for first row is less than frame_width
        if group.loc[0, 'frame_id'] != 0:
            while group.loc[0, 'start'] < frame_width:
                new_row = group.loc[0].copy()
                new_row['frame_id'] -= 1

                # Update 'tx' and 'start' for the new row
                new_row['tx'] = slices_df['tx'].loc[slices_df['frame_id'] == new_row['frame_id']]
                new_row['start'] += new_row['tx']

                if int(new_row['start']) >= frame_width:
                    break
                else:
                    group = pd.concat([pd.DataFrame(new_row).T, group]).reset_index(drop=True)

                    # Convert columns to integer type
                    for column in ['frame_id', 'tree_id', 'start', 'end']:
                        group[column] = group[column].astype(int)

        return group

    # Function to process start pixels for all groups
    def process_start_pixels(slices_df):
        return slices_df.groupby('tree_id').apply(set_start_pixels, slices_df=slices_df)

    # Function to set end pixels for each group
    def set_end_pixels(slices_df, slices_df_copy):
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

    # Call the function to process start pixels
    slices_df = process_start_pixels(slices_df)

    # Make a copy of slices dataframe
    slices_df_copy = slices_df.copy()

    # Call the function to set end pixels and get the updated dataframe
    updated_df = set_end_pixels(slices_df, slices_df_copy)

    # Reset the index and remove the 'tx' column
    updated_df.reset_index(drop=True, inplace=True)
    updated_df = updated_df.drop(columns=['tx'])

    # Save the updated dataframe to a csv file if output_path is provided
    if output_path is not None:
        updated_df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")

    return updated_df

if __name__ == "__main__":
    '''
    This script adds pixels coordinates to slices.csv file that was generated automatically by distance
    '''
    updated_df = process_slices_csv(path_slices_csv="slices.csv",
                                    path_translations_csv="jai_translations.csv",
                                    frame_width=1535,
                                    output_path="/home/lihi/FruitSpec/code/lihi/fsCounter/vision/trees_slicer/slice_by_distance/updated_df.csv")


    print('Data processing done')

    """
    The first and last slices are longer than the rest of the slices, since they are not centered around the center of the frame.
    
    """
