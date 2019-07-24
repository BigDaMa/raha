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
    "router",
    "state",
    "requester",
    "analytic",
    "text!mvc/template/navbar.template.html",
    "text!mvc/template/project.template.html",
    "mvc/HomeView"
], function (Router, State, Requester, Analytic, NavbarTemplate, ProjectTemplate, HomeView) {
    "use strict";
    function err(msg) {
        $('#projectModal-alert').html([
            ['<div class="alert alert-error">'],
            ['<button type="button" class="close" data-dismiss="alert">'],
            ['&times;</button>'],
            ['<span>' + msg + '</span></div>']
        ].join(''));
    }

    function triggerFilter(text) {
        var filter = State.get("filter");
        if (text !== filter) {
            State.set("filter", text);
            HomeView.refresh();
        }
    }

    function bindEvent() {
        $('#navbar').find("li a").on('click', function () {
            $('#navbar').find('.active').removeClass('active');
            $(this).addClass('active');
        });

        $('#project-modal-close').on('click', function () {
            var oldProject = State.get('project');
            if (oldProject !== null) {
                $('#projectModal').modal('hide');
            } else {
                err("No project is selected.");
            }
        });

        $("#refresh").on('click', HomeView.refresh);

        $("#analytic").on('click', function () {
            Analytic.start(State.get("project"));
        });

        $(document).on('keyup', function (e) {
            if (e.which === 27) {
                triggerFilter('');
                $("#searchbar").val('');
            }
        });

        $('#navbar-form').submit(function (e) {
            e.preventDefault();
            triggerFilter($("#searchbar").val());
        });

        $("#change").on("click", function () {
            $('#projectModal').find('.modal-body').remove();
            Requester.getProject({
                success: function (data) {
                    var projectList = _.flatten(data.data);
                    selectProject('projectModal', projectList);
                },
                failure: function () {
                    console.log("Getting project failed.");
                }
            });
        });

        $("#project-button").on("click", function () {
            var newProject = $("#create-new-project").val();
            if (newProject !== null && newProject !== "") {
                var pattern = new RegExp("^[a-zA-Z]\\w*");
                var match = pattern.exec(newProject);

                // regexp check failed.
                if (match && match[0] !== newProject) {
                    $("#project-input").addClass("error");
                    err("Input text has incorrect char.");
                    return;
                }
            }

            if (newProject !== null && newProject !== "") {
                Requester.createProject(newProject, {
                    success: function () {
                        $("#projectModal").modal('hide');
                        $("#projectName").text(newProject);
                        State.set('project', newProject);
                        Router.redirect('#home', { name : newProject });
                    },
                    failure: err
                });
            } else {
                var selectedProject = $("#select-existing-project").val();
                if (selectedProject === null) {
                    err("No project is selected.");
                    return;
                }

                enterProject(selectedProject);
            }
        });
    }

    function render() {
        $('body').append(_.template(NavbarTemplate)());
        $('#searchbar').val(State.get("filter"));
    }

    function selectProject(id, projectList) {
        var modalHtml = _.template(ProjectTemplate) ({projects: projectList});
        $('#' + id).find(".modal-footer").before(modalHtml);

        // event handler for modal
        $("#create-new-project").keypress(function () {
            $("#project-input").removeClass("error");
            $("#project-input").find("span").text("");
        });

        $('#' + id).modal('show');
    }

    function enterProject(selectedProject) {
        $("#projectModal").modal('hide');
        $("#projectName").text(selectedProject);
        State.set('project', selectedProject);
        Router.redirect('#home', { name : selectedProject });
    }

    function start() {
        render();
        bindEvent();

        // start project selection process
        Requester.getProject({
            success: function (data) {
                var projectList = _.flatten(data.data);
                var oldProject = State.get('project');
                if (oldProject && _.indexOf(projectList, oldProject) > -1) {
                    enterProject(oldProject);
                } else {
                    selectProject('projectModal', projectList);
                }
            },
            failure: function () {
                console.log("Getting project failed.");
            }
        });
    }

    return {
        start : start
    };
});
