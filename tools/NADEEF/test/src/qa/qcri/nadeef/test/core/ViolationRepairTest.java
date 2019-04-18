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

import com.google.common.collect.Lists;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import qa.qcri.nadeef.core.datamodel.*;
import qa.qcri.nadeef.core.utils.Bootstrap;
import qa.qcri.nadeef.core.utils.Violations;
import qa.qcri.nadeef.ruleext.FDRuleBuilder;
import qa.qcri.nadeef.test.NadeefTestBase;
import qa.qcri.nadeef.test.TestDataRepository;

import java.sql.Types;
import java.util.Collection;
import java.util.List;

/**
 * Violation repair test.
 */
@RunWith(Parameterized.class)
public class ViolationRepairTest extends NadeefTestBase {
    private Collection<Violation> violations;
    private FDRuleBuilder ruleBuilder;

    public ViolationRepairTest(String config_) {
        super(config_);
    }

    @Before
    public void setup() {
        try {
            Bootstrap.start(testConfig);
            violations = Violations.fromCSV(TestDataRepository.getViolationTestData1());
        } catch (Exception e) {
            Assert.fail("Setup failed.");
            e.printStackTrace();
        }
        ruleBuilder = new FDRuleBuilder();
    }

    @After
    public void tearDown() {
        Bootstrap.shutdown();
    }

    @Test
    public void FDRepair1() {
        try {
            Schema.Builder builder = new Schema.Builder();
            Schema schema =
                builder.table("test")
                    .column("B", Types.VARCHAR)
                    .column("A", Types.VARCHAR)
                    .column("C", Types.VARCHAR)
                    .build();
            List<Rule> rules =
                (List<Rule>) ruleBuilder.name("fd1")
                    .schema(schema)
                    .table("test")
                    .value("B|A,C")
                    .build();
            Rule rule = rules.get(0);
            List<Fix> allFixes = Lists.newArrayList();
            for (Violation violation : violations) {
                allFixes.addAll(rule.repair(violation));
            }
            Assert.assertEquals(2, allFixes.size());
            Fix fix2 = allFixes.get(0);
            Fix fix1 = allFixes.get(1);
            Cell left1 = fix1.getLeft();
            Cell right1 = fix1.getRight();
            Cell left2 = fix2.getLeft();
            Cell right2 = fix2.getRight();

            if (left1.getColumn().getColumnName().equals("c"))
                Assert.assertEquals("c", right1.getColumn().getColumnName());
            else if (left1.getColumn().getColumnName().equals("a"))
                Assert.assertEquals("a", right1.getColumn().getColumnName());

            if (left2.getColumn().getColumnName().equals("c"))
                Assert.assertEquals("c", right2.getColumn().getColumnName());
            else if (left2.getColumn().getColumnName().equals("a"))
                Assert.assertEquals("a", right2.getColumn().getColumnName());
        } catch (Exception e) {
            e.printStackTrace();
            Assert.fail(e.getMessage());
        }
    }
}
