import numpy as np
import pandas as pd
import re

import nflreadpy as nfl

class nflverse: 
    def __init__(self) -> None:
        self.seasons = list(range(2021, 2026))
        self.pbp_rows = ['play_id', 'game_id', 'down', 'ydstogo', 'first_down', 'play_type_nfl', 'touchdown', 'yardline_100', 'yards_gained', 'qtr', 'quarter_seconds_remaining', 'goal_to_go', 'score_differential', 'shotgun', 'no_huddle', 'fumble_lost', 'interception', 'season']
        self.participation_rows = ['play_id', 'nflverse_game_id', 'offense_formation', 'offense_personnel', 'defenders_in_box', 'defense_personnel', 'defense_man_zone_type', 'defense_coverage_type', 'number_of_pass_rushers']
        self.working_df = self._load_data_and_preprocess()
    
    
    def _parse_personnel(self, personnel, position_groups):
        """Convert personnel strings into counts for standardized position groups."""
        counts = {group: 0 for group in position_groups}

        if pd.isna(personnel):
            return pd.Series(counts)

        # Extract pairs such as ('4', 'DL'), ('3', 'CB')
        entries = re.findall(r'(\d+)\s*([A-Z]+)', personnel.upper())

        for count, position in entries:
            count = int(count)

            # Map each position to its standardized group
            for group, positions in position_groups.items():
                if position in positions:
                    counts[group] += count
                    break

        return pd.Series(counts)
    
    
    def _load_data_and_preprocess(self):
        pbp = nfl.load_pbp(self.seasons).to_pandas()
        pbp = pbp[self.pbp_rows]
 
        participation = nfl.load_participation(self.seasons).to_pandas()
        participation = participation[self.participation_rows]

        working_df = pd.merge(
            pbp,
            participation,
            left_on=['game_id', 'play_id'],
            right_on=['nflverse_game_id', 'play_id'],
            how='inner'
        )
        
        wanted_plays = ['PASS', 'RUSH', 'INTERCEPTION', 'SACK']
        working_df = working_df[working_df['play_type_nfl'].isin(wanted_plays)]
        working_df.loc[
            (working_df['play_type_nfl'] == 'INTERCEPTION') |  (working_df['play_type_nfl'] == 'SACK'),
            'play_type_nfl'
        ] = 'PASS'
        working_df['turnover'] = (working_df['fumble_lost'].astype(int) | working_df['interception'].astype(int)) # create turnover feature 
        categorical_cols_unknown = [
            'defense_man_zone_type',
            'defense_coverage_type',
            'offense_formation'
        ]

        working_df[categorical_cols_unknown] = working_df[categorical_cols_unknown].fillna('UNKNOWN')
        working_df['number_of_pass_rushers'] = working_df['number_of_pass_rushers'].fillna(-1)
        # Defensive personnel: DL, LB, DB
        defense_groups = {
            'def_DL': {'DL', 'DE', 'DT', 'NT'},
            'def_LB': {'LB', 'ILB', 'OLB', 'MLB'},
            'def_DB': {'DB', 'CB', 'FS', 'SS', 'S'}
        }

        # Offensive personnel: RB, TE, WR
        offense_groups = {
            'off_RB': {'RB', 'HB', 'FB', 'TB'},
            'off_TE': {'TE'},
            'off_WR': {'WR'}
        }

        # Generate standardized numerical features
        defense_counts = working_df['defense_personnel'].apply(
            self._parse_personnel, position_groups=defense_groups
        ).add_prefix('')

        offense_counts = working_df['offense_personnel'].apply(
            self._parse_personnel, position_groups=offense_groups
        ).add_prefix('')

        # Add features to the original dataframe
        working_df = pd.concat(
            [working_df, defense_counts, offense_counts],
            axis=1
        )
        
        return working_df

    def get_features_df(self):
        game_state = ['down', 'ydstogo', 'yardline_100', 'qtr', 'quarter_seconds_remaining', 'goal_to_go', 'score_differential']
        offensive_state = ['play_type_nfl', 'shotgun', 'no_huddle', 'offense_formation', 'off_RB', 'off_TE', 'off_WR']
        defensive_state = ['defenders_in_box', 'def_DL', 'def_LB', 'def_DB', 'defense_man_zone_type', 'defense_coverage_type', 'number_of_pass_rushers']
        X_features = game_state + offensive_state + defensive_state
        return self.working_df[X_features]
    
    def get_target_df(self):
        targets = ['first_down', 'touchdown', 'yards_gained', 'turnover']
        return self.working_df[targets]
    
    