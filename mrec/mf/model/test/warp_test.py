import unittest
import mrec.mf.model.warp as warp
import numpy
import warp_fast

class WarpTest(unittest.TestCase):
    def testCalEstimateWarpLoss(self):
        total_item_count = 100
        current_user_item_count = 20
        trials = 30

        warp_loss_dict = warp.WARP.precompute_warp_loss(total_item_count)
        estimated_warp_loss = warp.WARP.estimate_warp_loss_core(total_item_count, current_user_item_count, trials, warp_loss_dict)
        self.assertTrue(estimated_warp_loss - 1.8333333 < 0.0001)

        current_user_item_count = 60
        trials = 10
        estimated_warp_loss = warp.WARP.estimate_warp_loss_core(total_item_count, current_user_item_count, trials, warp_loss_dict)
        self.assertAlmostEqual(estimated_warp_loss, 2.08333333)

    def testApplyUpdate(self):
        to_update_vector = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0])
        delta_vector = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5])
        gamma = 30
        C = 0.01
        warp_fast.apply_update_core(to_update_vector, delta_vector, gamma, C)
        expect_array = numpy.array([0.00164677, 0.00288185, 0.00411694, 0.00535202, 0.0065871])
        for i in range(expect_array.shape[0]):
            self.assertAlmostEqual(to_update_vector[i], expect_array[i])
        gamma = 1
        C = 1
        warp_fast.apply_update_core(to_update_vector, delta_vector, gamma, C)
        expect_array = numpy.array([0.10164677, 0.20288185, 0.30411694, 0.40535202, 0.5065871])
        for i in range(expect_array.shape[0]):
            self.assertAlmostEqual(to_update_vector[i], expect_array[i])

    def testCalGradient(self):
        positive_item_factor = numpy.array([0.1, 0.2, 0.3, 0.4])
        negative_item_factor = numpy.array([0.9, 0.8, 0.7, 0.6])
        user_factor = numpy.array([0.6, 0.7, 0.8, 0.3])
        warp_loss = 1.8333

        dU, dV_pos, dV_nega = warp.WARPDecomposition.compute_gradient_step_core(positive_item_factor, negative_item_factor,
                                                                                user_factor, warp_loss)
        print dU
        print dV_pos
        print dV_nega
        expect_array = numpy.array([-1.46664, -1.09998, -0.73332, -0.36666])
        for i in range(expect_array.shape[0]):
            self.assertAlmostEqual(dU[i], expect_array[i])

        expect_array = numpy.array([1.09998, 1.28331, 1.46664, 0.54999])
        for i in range(expect_array.shape[0]):
            self.assertAlmostEqual(dV_pos[i], expect_array[i])

        expect_array = numpy.array([-1.09998, -1.28331, -1.46664, -0.54999])
        for i in range(expect_array.shape[0]):
            self.assertAlmostEqual(dV_nega[i], expect_array[i])

if __name__ == '__main__':
    unittest.main()
