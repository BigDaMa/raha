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

@RunWith(Parameterized.class)
public class DCDetectionTest extends NadeefTestBase {
    public DCDetectionTest(String testConfig_) {
        super(testConfig_);
    }

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);
            NadeefConfiguration.setMaxIterationNumber(1);
            NadeefConfiguration.setAlwaysOverride(true);
            DBInstaller.uninstall(NadeefConfiguration.getDbConfig());
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail(ex.getMessage());
        }
    }

    @After
    public void teardown() {
        Bootstrap.shutdown();
    }

    @Test
    public void cleanExecutorConstantDCTest(){
        try{
            CleanPlan cleanPlan = TestDataRepository.getConstantDCTestPlan();
            CleanExecutor executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(5);
        } catch (Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void cleanExecutorDCTest(){
        CleanExecutor executor = null;
        try{
            CleanPlan cleanPlan = TestDataRepository.getDCTestPlan();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(48);
        } catch (Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        } finally {
            if (executor != null) {
                executor.shutdown();
            }
        }
    }

    @Test
    public void cleanExecutorSingleDCTest(){
        CleanExecutor executor = null;
        try{
            CleanPlan cleanPlan = TestDataRepository.getSingleTupleDCTestPlan();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(12);
        } catch (Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        } finally {
            if (executor != null) {
                executor.shutdown();
            }
        }
    }

    @Test
    public void cleanExecutorFloatDCTest() {
        CleanExecutor executor = null;
        try{
            CleanPlan cleanPlan = TestDataRepository.getFloatDCTestPlan();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(6);
        } catch(Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        } finally {
            if (executor != null) {
                executor.shutdown();
            }
        }
    }

    @Test
    public void dcGeneratedFileTest() {
        CleanExecutor executor = null;
        try{
            CleanPlan cleanPlan = TestDataRepository.getDCGeneratedCleanPlan();
            executor = new CleanExecutor(cleanPlan);
            executor.detect();
            verifyViolationResult(5);
        } catch(Exception e){
            e.printStackTrace();
            Assert.fail(e.getMessage());
        } finally {
            if (executor != null) {
                executor.shutdown();
            }
        }
    }

    private void verifyViolationResult(int expectRow)
        throws Exception {
        int rowCount = Violations.getViolationRowCount(NadeefConfiguration.getDbConfig());
        Assert.assertEquals(expectRow, rowCount);
    }
}
