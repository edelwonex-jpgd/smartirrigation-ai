import unittest
from model import make_synthetic_dataset, train_model, predict_one, estimate_water_deficit_mm, FEATURES

class TestSmartIrrigation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=make_synthetic_dataset(n=500, seed=42)
        cls.model,cls.stats=train_model(cls.df)

    def test_dataset(self):
        self.assertEqual(len(self.df),500)
        self.assertEqual(set(FEATURES+['target']),set(self.df.columns))

    def test_prediction(self):
        row=dict(soil_moisture_pct=23,field_capacity_pct=35,wilting_point_pct=15,
                 temperature_c=30,forecast_rain_mm=5,root_depth_cm=30,crop_stage='Berbunga')
        result=predict_one(self.model,row)
        self.assertIn(result['status'],['Belum perlu','Pantau','Perlu irigasi'])
        self.assertAlmostEqual(result['net_deficit_mm'],36)
        self.assertAlmostEqual(result['gross_depth_mm'],42.4)

    def test_invalid(self):
        row=dict(soil_moisture_pct=23,field_capacity_pct=10,wilting_point_pct=15,
                 temperature_c=30,forecast_rain_mm=5,root_depth_cm=30,crop_stage='Berbunga')
        with self.assertRaises(ValueError):
            estimate_water_deficit_mm(row)

if __name__ == '__main__':
    unittest.main()
