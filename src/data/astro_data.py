import pandas as pd
import numpy as np
from astropy.coordinates import solar_system_ephemeris, get_body, EarthLocation
from astropy.time import Time
from datetime import datetime, timedelta
from loguru import logger

class AstroData:
    """占星数据获取类"""
    
    def __init__(self):
        """初始化"""
        self.planets = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto']
    
    def get_planet_positions(self, date):
        """
        获取指定日期的行星位置
        
        Args:
            date: 日期，格式为 YYYY-MM-DD
            
        Returns:
            dict: 行星位置数据
        """
        try:
            # 转换日期格式
            time = Time(date)
            
            # 获取行星位置
            positions = {}
            with solar_system_ephemeris.set('builtin'):
                for planet in self.planets:
                    try:
                        # 获取行星位置（地心坐标系）
                        planet_pos = get_body(planet, time, EarthLocation.of_site('greenwich'))
                        # 转换为赤道坐标系
                        ra, dec = planet_pos.ra.deg, planet_pos.dec.deg
                        positions[planet] = {'ra': ra, 'dec': dec}
                    except Exception as e:
                        logger.warning(f"获取{planet}位置失败: {e}")
                        positions[planet] = {'ra': np.nan, 'dec': np.nan}
            
            return positions
        except Exception as e:
            logger.error(f"获取行星位置失败: {e}")
            return {}
    
    def get_planetary_aspects(self, date):
        """
        获取指定日期的行星相位
        
        Args:
            date: 日期，格式为 YYYY-MM-DD
            
        Returns:
            dict: 行星相位数据
        """
        try:
            # 获取行星位置
            positions = self.get_planet_positions(date)
            
            # 计算相位
            aspects = {}
            for i, planet1 in enumerate(self.planets):
                for j, planet2 in enumerate(self.planets):
                    if i < j:
                        # 计算角度差
                        ra1 = positions[planet1]['ra']
                        ra2 = positions[planet2]['ra']
                        if not np.isnan(ra1) and not np.isnan(ra2):
                            angle = abs(ra1 - ra2) % 360
                            # 标准化到 0-180 度
                            angle = min(angle, 360 - angle)
                            aspects[f"{planet1}_{planet2}"] = angle
            
            return aspects
        except Exception as e:
            logger.error(f"获取行星相位失败: {e}")
            return {}
    
    def get_astro_data(self, start_date, end_date):
        """
        获取指定日期范围内的占星数据
        
        Args:
            start_date: 开始日期，格式为 YYYY-MM-DD
            end_date: 结束日期，格式为 YYYY-MM-DD
            
        Returns:
            DataFrame: 占星数据
        """
        try:
            # 生成日期范围
            date_range = pd.date_range(start=start_date, end=end_date)
            
            # 初始化数据列表
            data = []
            
            # 遍历日期
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                logger.info(f"获取日期 {date_str} 的占星数据")
                
                # 获取行星位置
                positions = self.get_planet_positions(date_str)
                
                # 获取行星相位
                aspects = self.get_planetary_aspects(date_str)
                
                # 构建数据行
                row = {'date': date_str}
                
                # 添加行星位置数据
                for planet, pos in positions.items():
                    row[f"{planet}_ra"] = pos['ra']
                    row[f"{planet}_dec"] = pos['dec']
                
                # 添加行星相位数据
                for aspect, angle in aspects.items():
                    row[aspect] = angle
                
                data.append(row)
            
            # 转换为 DataFrame
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            logger.error(f"获取占星数据失败: {e}")
            return pd.DataFrame()
