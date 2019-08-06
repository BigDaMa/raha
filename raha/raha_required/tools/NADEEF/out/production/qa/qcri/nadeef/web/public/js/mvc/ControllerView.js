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

define([
    'requester',
    'table',
    'jquery.filedrop',
    'state',
    'ruleminer',
    'analytic',
    'mvc/RuleEditorView',
    'mvc/ProgressbarView',
    'mvc/SourceEditorView',
    'text!mvc/template/controller.template.html',
    'text!mvc/template/detail.template.html'
], function (
    Requester,
    Table,
    FileDrop,
    State,
    RuleMiner,
    Analytic,
    RuleEditorView,
    ProgressBarView,
    SourceEditorView,
    ControllerTemplate,
    DetailTemplate) {
    "use strict";
    var domId;

    function info(msg) {
        $('#home-alert-info').alert('close');
        $('#home-alert').html([
            ['<div class="alert alert-success" id="home-alert-info">'],
            ['<button type="button" class="close" data-dismiss="alert">'],
            ['&times;</button>'],
            ['<span><h4>' + msg + '</h4></span></div>']
        ].join(''));

        window.setTimeout(function () { $('#home-alert-info').alert('close'); }, 3000);
    }

    function err(msg) {
        if (_.isObject(msg) && 'responseText' in msg) {
            var json = JSON.parse(msg.responseText);
            msg = json.error;
        }

        $('#home-alert').html([
            ['<div class="alert alert-error">'],
            ['<button type="button" class="close" data-dismiss="alert">'],
            ['&times;</button>'],
            ['<span><h4>' + msg + '</h4></span></div>']
        ].join(''));
    }

    function render(id) {
        domId = id;

        refresh();

        window.addEventListener("message", function (event) {
            if (event.origin.indexOf("ruleminer") > -1 && event.data.indexOf("inserted") > -1) {
                console.log('received: ' + event.data);
                refreshRuleList();
            }
        }, false);

        window.addEventListener('beforeunload', function () {
            RuleMiner.close();
            Analytic.close();
        });

        $('#' + domId)[0].addEventListener('refreshRuleEvent', function (e) {
            console.log("refreshRule event received.");
            if ("detail" in e && "info" in e.detail) {
                info(e.detail.info);
            }
            refreshRuleList();
        });

        $('#' + domId)[0].addEventListener('refreshSource', function () {
            console.log("refreshSource event received.");
            refreshSourceList();
        });
    }

    function refresh() {
        refreshSourceList();
        refreshRuleList();
    }

    function refreshSourceList() {
        console.log("Refresh source list");
        Requester.getSource({
            success: function (source) {
                var sources = source.data;
                State.set('source', sources);
                var html =
                    _.template(ControllerTemplate)({ sources: sources});
                $('#' + domId).html(html);
                $('#refresh_source').on('click', function () {
                    refresh();
                });

                $('#selected_source').on('change', function () {
                    var newSource = $("#selected_source").val();
                    State.set("currentSource", newSource);
                    renderRuleList(newSource);
                    Table.load({ domId: 'source-table', table: newSource });
                });

                $('#new-source').on('click', function () {
                    $('#source-editor').modal('show');
                });

                $('#btn-discover').on('click', function () {
                    if (RuleMiner.isAvailable()) {
                        if (RuleMiner.isWindowOpened()) {
                            info("Rule Miner window is already opened.");
                        } else {
                            var project = State.get("project");
                            var table = State.get("currentSource");
                            if (_.isNull(project) || _.isNull(table)) {
                                err("No table is selected.");
                                return false;
                            }
                            if (!RuleMiner.start(project, table)) {
                                err("Starting Rule Miner failed.");
                            }
                        }
                    } else {
                        err("Rule Miner is not available.");
                    }
                });

                $('#new_plan').on('click', function () {
                    var editor = new RuleEditorView.Create($('#rule-editor-modal')[0], {
                        name: null,
                        type: 'FD',
                        table1: State.get("currentSource")
                    });
                    editor.render();
                });

                $('#edit_plan').on('click', function () {
                    var selectedRule = State.get('currentRule');
                    if (selectedRule == null) {
                        err('No rule is selected.');
                        return;
                    }

                    if (_.isArray(selectedRule) && selectedRule.length !== 1) {
                        err('Can not edit multiple rules.');
                        return;
                    }

                    Requester.getRuleDetail(selectedRule[0], {
                        success: function (data) {
                            var plan = data['data'][0];
                            var editor = new RuleEditorView.Create(
                                $('#rule-editor-modal')[0],
                                arrayToPlan(plan)
                            );
                            editor.render();
                        }
                    });
                });

                $('#selected_rule').on('change', function () {
                    var newRule = $("#selected_rule").val();
                    State.set('currentRule', newRule);
                    renderRuleDetail();
                    // only reload the violation table when the violation tab is active
                    var activeTable = $('#tables .tab-content').find('div.active')[0].id;
                    if (activeTable.indexOf("violation") > -1) {
                        Table.load({ domId: 'violation-table', table: "violation", rule: newRule });
                    }
                });

                $('#detect').on('click', function () {
                    var selectedPlan = State.get("currentRule");
                    if (selectedPlan == null || !_.isArray(selectedPlan)) {
                        err('No rule is selected.');
                        return;
                    }

                    detect(selectedPlan);
                });

                $('#delete').on('click', function () {
                    var selectedRule = State.get("currentRule");
                    if (selectedRule == null || !_.isArray(selectedRule)) {
                        err('No rule is selected.');
                        return;
                    }

                    Requester.deleteRule({ rules: selectedRule }, {
                        success: function () {
                            info("Selected rules are deleted.");
                            refreshRuleList();
                        },
                        failure: err
                    });
                });

                State.clear("currentSource");
            },
            failure: err,
            always: function () {
                // TODO: to move
                if ($("#progressbar-content").length === 0) {
                    ProgressBarView.start('progressbar');
                }

                if ($("#source-editor").length === 0) {
                    SourceEditorView.start('source-editor-modal');
                }
            }
        });
    }

    function refreshRuleList() {
        console.log("Refresh rule list");
        Requester.getRule({
            success: function (data) {
                State.set('rule', data.data);
                if (State.get("currentSource")) {
                    renderRuleList(State.get("currentSource"));
                }
            },
            failure: err
        });
    }

    function arrayToPlan(v) {
        return {
            name: v[0],
            type: v[1],
            code: v[2],
            table1: v[3],
            table2: v[4]
        };
    }

    function renderRuleDetail() {
        var selectedRule = State.get('currentRule');
        if (_.isUndefined(selectedRule) || _.isNull(selectedRule)) {
            return;
        }

        if (_.isArray(selectedRule) && selectedRule.length !== 1) {
            $('#detail').html('');
        } else {
            Requester.getRuleDetail(
                selectedRule, {
                success: function (data) {
                    var plan = arrayToPlan(data['data'][0]);
                    $('#detail').html(_.template(DetailTemplate, plan));
                }
            });
        }
    }

    function renderRuleList(source) {
        var ruleList = State.get('rule');
        var selectedRule =
            _.filter(ruleList, function (x) { return x[3] === source || x[4] === source; });
        var selectedHtml = _.template(
            "<% _.each(rules, function(rule) { %>" +
            "<option value='<%= rule[0] %>'><%= rule[0] %></option>" +
            "<% }); %>", { rules : selectedRule });
        $('#selected_rule').html(selectedHtml);
        State.clear("currentRule");
    }

    function detect(plans) {
        // clean the violation before start new detection.
        $.when(Requester.deleteViolation()).then(
            function () {
                _.each(plans, function (planName) {
                    Requester.getRuleDetail(planName, {
                        success: function (data) {
                            var plan = arrayToPlan(data['data'][0]);
                            Requester.doDetect(
                                plan,
                                {
                                    success: function (data) {
                                        info("A job is successfully submitted.");
                                        var key = data.data;
                                        console.log('Received job key : ' + key);
                                        if (key != null) {
                                            var jobList = State.get("job");
                                            jobList.push({ name : plan.name, key : key });
                                            State.set("job", jobList);
                                        }
                                    },
                                    failure: err
                                }
                            );
                        },
                        failure: err
                    });
                });
            });
    }

    return {
        render: render
    };
});
