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

package qa.qcri.nadeef.test.rulebuilder;

import com.google.common.io.Files;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.datamodel.Schema;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.ruleext.FDRuleBuilder;
import qa.qcri.nadeef.test.NadeefTestBase;

import java.io.File;
import java.sql.Types;
import java.util.Collection;

/**
 * Test for FD Rule writer.
 */
@RunWith(Parameterized.class)
public class FDRuleBuilderTest extends NadeefTestBase {
    private static File workingDirectory;

    public FDRuleBuilderTest(String testConfig_) {
        super(testConfig_);
    }

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);
            workingDirectory = Files.createTempDir();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    @After
    public void tearDown() {
        Bootstrap.shutdown();
        // remove all the temp files.
        workingDirectory.delete();
    }

    @Test
    public void testFDFileGeneration() {
        FDRuleBuilder fdRuleBuilder = new FDRuleBuilder();
        File output = null;
        try {
            Schema.Builder schemaBuilder = new Schema.Builder();
            Schema schema =
                schemaBuilder
                    .table("table")
                    .column("A", Types.VARCHAR)
                    .column("B", Types.VARCHAR)
                    .column("C", Types.VARCHAR)
                    .build();
            output = fdRuleBuilder
                .name("Test")
                .table("table")
                .schema(schema)
                .out(workingDirectory)
                .value("A|B, C")
                .compile().iterator().next();
            System.out.println("Write file in " + output.getAbsolutePath());
            Assert.assertTrue(output.exists());

            schema =
                new Schema.Builder()
                    .table("table")
                    .column("D", Types.VARCHAR)
                    .column("E", Types.VARCHAR)
                    .column("F", Types.VARCHAR)
                    .column("GG", Types.VARCHAR)
                    .build();
            output = fdRuleBuilder
                .table("table")
                .schema(schema)
                .out(workingDirectory)
                .value("D,E|F, GG")
                .compile().iterator().next();
            System.out.println("Write file in " + output.getAbsolutePath());
            Assert.assertTrue(output.exists());
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testFDLoad() {
        FDRuleBuilder fdRuleBuilder = new FDRuleBuilder();
        Collection<Rule> output = null;
        try {
            Schema.Builder schemaBuilder = new Schema.Builder();
            Schema schema =
                schemaBuilder
                    .table("table")
                    .column("A", Types.VARCHAR)
                    .column("B", Types.VARCHAR)
                    .column("C", Types.VARCHAR)
                    .build();
            output = fdRuleBuilder
                .name("Test")
                .schema(schema)
                .table("table")
                .value("A|B, C")
                .build();
            Assert.assertTrue(output.size() == 1);
            Assert.assertTrue(output.iterator().next() != null);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }

    @Test
    public void testFDLoad2() {
        FDRuleBuilder fdRuleBuilder = new FDRuleBuilder();
        Collection<Rule> output = null;
        try {
            Schema.Builder schemaBuilder = new Schema.Builder();
            Schema schema =
                schemaBuilder
                    .table("table")
                    .column("A", Types.VARCHAR)
                    .column("B", Types.VARCHAR)
                    .column("C", Types.VARCHAR)
                    .build();
            output = fdRuleBuilder
                .table("table")
                .schema(schema)
                .value("A|B, C")
                .build();
            Assert.assertTrue(output.size() == 1);
            Assert.assertTrue(output.iterator().next() != null);
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }
}
