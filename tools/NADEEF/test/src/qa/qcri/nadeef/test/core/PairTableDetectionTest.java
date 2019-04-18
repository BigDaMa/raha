/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.test.core;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.pipeline.CleanExecutor;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.Violations;
import qa.qcri.nadeef.core.utils.sql.DBInstaller;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;

/**
 * Pair table unit test.
 */
@RunWith(Parameterized.class)
public class PairTableDetectionTest extends NadeefTestBase {
    public PairTableDetectionTest(String config_) {
        super(config_);
    }

    @Before
    public void setUp() {
        try {
            Bootstrap.start(testConfig);
            NadeefConfiguration.setAlwaysOverride(true);

            DBInstaller.uninstall(NadeefConfiguration.getDbConfig());
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    @After
    public void teardown() {
        Bootstrap.shutdown();
    }

    @Test
    public void pairCleanPlanTest1() {
        try {
            CleanPlan cleanPlan = TestDataRepository.getPairCleanPlan1();
            CleanExecutor executor = new CleanExecutor(cleanPlan);
            executor.detect();
            int rowCount =
                Violations.getViolationRowCount(NadeefConfiguration.getDbConfig());
            Assert.assertEquals(10, rowCount);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }
}
