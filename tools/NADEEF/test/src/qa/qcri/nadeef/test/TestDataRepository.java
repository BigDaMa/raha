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

package qa.qcri.nadeef.test;

import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;

import java.io.File;
import java.io.FileReader;
import java.util.List;

/**
 * Factory class to get TestData.
 */
public class TestDataRepository {
    public static String DerbyConfig =
        "test/src/qa/qcri/nadeef/test/input/config/derbyConfig.conf";
    public static String PostgresConfig =
        "test/src/qa/qcri/nadeef/test/input/config/postgresConfig.conf";
    public static String MySQLConfig =
            "test/src/qa/qcri/nadeef/test/input/config/mysqlConfig.conf";

    public static File getDumpTestCSVFile() {
        return new File("test/src/qa/qcri/nadeef/test/input/dumptest.csv");
    }

    public static File getPairCleanPlanFile1() {
        return new File("test/src/qa/qcri/nadeef/test/input/PairTableCleanPlan1.json");
    }

    public static File getFailurePlanFile1() {
        return new File("test/src/qa/qcri/nadeef/test/input/FailurePlan1.json");
    }

    public static File getFailurePlanFile2() {
        return new File("test/src/qa/qcri/nadeef/test/input/FailurePlan2.json");
    }

    public static File getTestFile1() {
        return new File("test/src/qa/qcri/nadeef/test/input/CleanPlan1.json");
    }

    public static File getTestFile2() {
        return new File("test/src/qa/qcri/nadeef/test/input/CleanPlan2.json");
    }

    public static File getTestFile3() {
        return new File("test/src/qa/qcri/nadeef/test/input/CleanPlan3.json");
    }

    public static File getTestFile4() {
        return new File("test/src/qa/qcri/nadeef/test/input/CleanPlan4.json");
    }

    public static File getTestFile5() {
        return new File("test/src/qa/qcri/nadeef/test/input/CleanPlan5.json");
    }

    public static File getViolationTestData1() {
        return new File("test/src/qa/qcri/nadeef/test/input/violation1.csv");
    }

    public static File getFixTestData1() {
        return new File("test/src/qa/qcri/nadeef/test/input/Fix1.csv");
    }

    public static File getLocationData1() {
        return new File("test/src/qa/qcri/nadeef/test/input/locations.csv");
    }

    public static File getCleanPlan6File() {
        return new File("test/src/qa/qcri/nadeef/test/input/CleanPlan6.json");
    }

    public static File getCleanPlan7File() {
        return new File("test/src/qa/qcri/nadeef/test/input/CleanPlan7.json");
    }

    public static File getCleanPlan8File() {
        return new File("test/src/qa/qcri/nadeef/test/input/CleanPlan8.json");
    }

    public static File getCleanPlan9File() {
        return new File("test/src/qa/qcri/nadeef/test/input/CleanPlan9.json");
    }

    public static File getHolisticPlan1File() {
        return new File("test/src/qa/qcri/nadeef/test/input/HolisticTestPlan1.json");
    }

    public static File getStressPlan10kFile() {
        return new File("test/src/qa/qcri/nadeef/test/input/StressPlan10k.json");
    }

    public static File getStressPlan30kFile() {
        return new File("test/src/qa/qcri/nadeef/test/input/StressPlan30k.json");
    }

    public static File getStressPlan40kFile() {
        return new File("test/src/qa/qcri/nadeef/test/input/StressPlan40k.json");
    }

    public static File getStressPlan80kFile() {
        return new File("test/src/qa/qcri/nadeef/test/input/StressPlan80k.json");
    }

    public static List<CleanPlan> getHolisticTestPlan1()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getHolisticPlan1File()), NadeefConfiguration.getDbConfig());
    }

    public static File getDCTestFile(){
        return new File("test/src/qa/qcri/nadeef/test/input/DCCleanPlan3.json");
    }

    public static File getConstantDCTestFile(){
        return new File("test/src/qa/qcri/nadeef/test/input/DCCleanPlan2.json");
    }

    public static File getSingleTupleDCTestFile(){
        return new File("test/src/qa/qcri/nadeef/test/input/DCCleanPlan1.json");
    }

    public static File getFloatDCTestFile(){
        return new File("test/src/qa/qcri/nadeef/test/input/DCCleanPlan4.json");
    }

    public static File getDCGeneratedFile(){
        return new File("test/src/qa/qcri/nadeef/test/input/DCCleanPlan5.json");
    }

    public static File getIncCleanPlanFile() {
        return new File("test/src/qa/qcri/nadeef/test/input/IncCleanPlan1.json");
    }

    public static File getIncCleanPlanFile2() {
        return new File("test/src/qa/qcri/nadeef/test/input/IncCleanPlan2.json");
    }

    public static CleanPlan getCleanPlan()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getTestFile1()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getCleanPlan2()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getTestFile2()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getCleanPlan3()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getTestFile3()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getCleanPlan4()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getTestFile4()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static List<CleanPlan> getCleanPlan5()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getTestFile5()), NadeefConfiguration.getDbConfig());
    }

    public static CleanPlan getCleanPlan6()
        throws Exception {
            return CleanPlan.create(
                new FileReader(getCleanPlan6File()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getCleanPlan7()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getCleanPlan7File()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getCleanPlan8()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getCleanPlan8File()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getCleanPlan9()
        throws Exception {

        return CleanPlan.create(
            new FileReader(getCleanPlan9File()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getStressPlan10k() throws Exception {
        return CleanPlan.create(
            new FileReader(getStressPlan10kFile()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getStressPlan30k()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getStressPlan30kFile()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getStressPlan40k()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getStressPlan40kFile()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getStressPlan80k()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getStressPlan80kFile()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getPairCleanPlan1()
        throws Exception {
        return CleanPlan.create(
            new FileReader(getPairCleanPlanFile1()), NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getDCTestPlan()
        throws Exception {
      return CleanPlan.create(
          new FileReader(getDCTestFile()),
          NadeefConfiguration.getDbConfig()
      ).get(0);
    }

    public static CleanPlan getConstantDCTestPlan()
        throws Exception {
      return CleanPlan.create(
          new FileReader(getConstantDCTestFile()),
          NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getSingleTupleDCTestPlan()
        throws Exception {

      return CleanPlan.create(
          new FileReader(getSingleTupleDCTestFile()),
          NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getFloatDCTestPlan() throws Exception {
      return CleanPlan.create(
          new FileReader(getFloatDCTestFile()),
          NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getDCGeneratedCleanPlan() throws Exception {
        return CleanPlan.create(
            new FileReader(getDCGeneratedFile()),
            NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getIncCleanPlan1() throws Exception {
        return CleanPlan.create(
            new FileReader(getIncCleanPlanFile()),
            NadeefConfiguration.getDbConfig()).get(0);
    }

    public static CleanPlan getIncCleanPlan2() throws Exception {
        return CleanPlan.create(
            new FileReader(getIncCleanPlanFile2()),
            NadeefConfiguration.getDbConfig()).get(0);
    }

    public static List<CleanPlan> getPlan(String relativeFileName) throws Exception {
        File inputFile = new File("test/src/qa/qcri/nadeef/test/input/" + relativeFileName);
        return CleanPlan.create(new FileReader(inputFile), NadeefConfiguration.getDbConfig());
    }
}

